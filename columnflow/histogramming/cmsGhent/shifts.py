# coding: utf-8

"""
Shift collections: one :py:class:`~columnflow.tasks.histograms.CreateHistograms` job fills all subshifts of a collection
shift (``<name>_{up,down}``, see :py:func:`add_shift_collection`) with the hist producer :py:func:`shifts_hist`.

Weight-only subshifts are filled from the nominal arrays with their varied event weight. Selection-dependent subshifts
(tag ``selection_dependent``) are filled from the reduced events plus their own ProduceColumns (and MLEvaluation)
outputs, merged and aliased as for a ``--shift <subshift>`` job.
"""

from __future__ import annotations

import contextlib

import law
import order as od

from columnflow.histogramming import HistProducer, hist_producer
from columnflow.util import maybe_import, DotDict
from columnflow.hist_util import create_hist_from_variables, fill_hist, translate_hist_intcat_to_strcat
from columnflow.columnar_util import has_ak_column, Route, optional_column as optional, ChunkedParquetReader
from columnflow.types import Any, Iterable

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)

# tag of a shift collection, the names of its subshifts are in the auxiliary "subshifts"
COLLECTION_TAG = "is_collection"

# tag of a subshift whose effect is not only an event weight (filled from its own ProduceColumns outputs)
SELECTION_DEPENDENT_TAG = "selection_dependent"

# column with the global entry index of each event in the reduced events file (written by shifts_hist_setup); it tells
# the hist producer which entries of the subshifts' ProduceColumns outputs belong to the current chunk
ROW_COLUMN = "shifts_hist_row"

# CreateHistograms methods (change Ghent) that apply the steps of CreateHistograms.run to the shifted events
TASK_METHODS = ("merge_chunk_columns", "get_category_ids", "get_fill_data")


def add_shift_collection(
    config: od.Config,
    name: str,
    id_offset: int,
    shift_names: Iterable[str],
    weight_column: str | None = None,
) -> list[od.Shift]:
    """
    Adds the shift collections ``<name>_up`` and ``<name>_down`` to *config* (type ``shape``, tag ``is_collection``,
    ids counted up from *id_offset*: + 1 for the first one added, + 2 for the second) and returns the added shifts.
    Collections that already exist are skipped.

    The auxiliary ``subshifts`` of a collection lists the names of the config shifts that match one of the patterns in
    *shift_names* (:py:func:`law.util.multi_match`) and end with the direction of the collection, in the order of
    ``config.shifts``. Add the subshifts to the config before calling this function.

    The collections must be known to the hist producer of the CreateHistograms job, i.e. be a shift that one of the event
    weights depends on. With *weight_column* (e.g. ``"normalization_weight"``), the added collections are appended to the
    shifts of that entry of ``config.x.event_weights``, which must exist.
    """
    shift_names = list(shift_names)
    shift_insts = []
    idx = id_offset
    for direction in [od.Shift.UP, od.Shift.DOWN]:
        col_name = f"{name}_{direction}"
        if config.has_shift(col_name):
            continue
        subshift_insts = [
            s for s in config.shifts
            if law.util.multi_match(s.name, shift_names)
            if s.name.endswith(direction)
        ]
        shift_insts.append(config.add_shift(
            col_name,
            id=(idx := idx + 1),
            type="shape",
            tags=[COLLECTION_TAG],
            aux=dict(subshifts=[s.name for s in subshift_insts]),
        ))

    if weight_column is not None:
        config.x.event_weights[weight_column] += shift_insts

    return shift_insts


def event_weight(self: HistProducer, events: ak.Array) -> ak.Array:
    """
    Full event weight as in the ``default`` hist producer: MC ∏ ``cfg.x.event_weights`` × ∏ ``dataset.x.event_weights``,
    data ∏ ``cfg.x.data_event_weights``.
    """
    weight = ak.Array(np.ones(len(events)))
    if not len(events):
        return weight

    if self.dataset_inst.is_mc:
        for column in self.config_inst.x.event_weights:
            weight = weight * Route(column).apply(events)
        for column in self.dataset_inst.x("event_weights", []):
            if has_ak_column(events, column):
                weight = weight * Route(column).apply(events)
            else:
                logger.warning_once(
                    f"missing_dataset_weight_{column}",
                    f"weight '{column}' for dataset {self.dataset_inst.name} not found",
                )
    elif self.config_inst.x.data_event_weights:
        for column in self.config_inst.x.data_event_weights:
            weight = weight * Route(column).apply(events)

    return weight


def set_subshift_insts(self: HistProducer, task: law.Task) -> None:
    """
    Sorts the subshifts of the task's shift collection (none for a normal shift) into weight-only subshifts (filled from
    the nominal arrays) and selection-dependent subshifts (filled from their own ProduceColumns outputs). Weight-only
    subshifts whose aliased columns are no event weight of the config or of any dataset would only repeat the nominal
    histogram and are skipped with a warning.
    """
    config_inst = self.config_inst
    subshift_insts = sorted(
        (config_inst.get_shift(s) for s in task.local_shift_inst.x("subshifts", []) if config_inst.has_shift(s)),
        key=lambda shift_inst: shift_inst.id,
    )
    if not self.dataset_inst.is_mc:
        subshift_insts = []

    self.sel_subshift_insts = [s for s in subshift_insts if SELECTION_DEPENDENT_TAG in s.tags]

    weight_columns = set(config_inst.x("event_weights", {}))
    for dataset_inst in config_inst.datasets:
        weight_columns |= set(dataset_inst.x("event_weights", {}))
    self.weight_subshift_insts = []
    skipped = []
    for subshift_inst in subshift_insts:
        if subshift_inst in self.sel_subshift_insts:
            continue
        if weight_columns & set(subshift_inst.x("column_aliases", {})):
            self.weight_subshift_insts.append(subshift_inst)
        else:
            skipped.append(subshift_inst.name)
    if skipped:
        logger.warning_once(
            f"shifts_hist_skipped_{task.local_shift_inst.name}",
            f"shifts_hist: skipping subshifts of {task.local_shift_inst.name} whose columns are no event weight: "
            f"{', '.join(skipped)}",
        )

    self.subshift_insts = self.weight_subshift_insts + self.sel_subshift_insts


def is_collection_job(self: HistProducer, task: law.Task) -> bool:
    """
    True when *task* processes a shift collection (its local shift has subshifts that are filled here).
    """
    return bool(self.subshift_insts) and task.shift == task.global_shift_inst.name


def clean_fill_data(data: dict[str, Any], variable_insts: list[od.Variable]) -> dict[str, Any]:
    """
    Removes entries with nan or None in any variable from *data* and converts discrete variables to int.
    """
    nan_masks = ak.all([
        ~(ak.is_none(ak.nan_to_none(data[variable_inst.name])))
        for variable_inst in variable_insts
    ], axis=0)

    new_data = {}
    for key, value in data.items():
        if isinstance(value, (ak.Array, np.ndarray)):
            new_data[key] = value[nan_masks]
        else:
            new_data[key] = value

    for variable_inst in variable_insts:
        if variable_inst.discrete_x:
            new_data[variable_inst.name] = ak.to_numpy(new_data[variable_inst.name]).astype(int)

    return new_data


@hist_producer()
def shifts_hist(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Event weight as the ``default`` hist producer. On a shift collection (tag ``is_collection``, the subshifts are in
    ``shift.x.subshifts``, see :py:func:`add_shift_collection`) it fills one histogram per subshift:

    - weight-only subshifts from the nominal arrays: the returned weight has one column per subshift (nominal weight ×
      variation / nominal of the aliased weight columns),
    - selection-dependent subshifts (tag ``selection_dependent``, e.g. JEC, JER or lepton energy scale and resolution)
      from the reduced events plus their own ProduceColumns (and MLEvaluation) outputs, merged and aliased as
      CreateHistograms does for a ``--shift <subshift>`` job; the shifted events, weights and category ids of the
      current chunk are kept for ``shifts_hist_fill_hist``.

    The collection itself (no ``nominal`` entry) is not filled. On any other shift it behaves as ``default``.
    """
    task = kwargs["task"]

    weight = event_weight(self, events)
    self.subshift_chunk = None

    if not (is_collection_job(self, task) and len(events)):
        return events, weight

    if self.sel_subshift_insts:
        self.subshift_chunk = load_subshift_chunk(self, events, task)

    if not self.weight_subshift_insts:
        return events, weight

    event_weights = set(self.config_inst.x("event_weights", {})) | set(self.dataset_inst.x("event_weights", {}))
    all_weights = []
    for subshift_inst in self.weight_subshift_insts:
        subshift_weight = weight
        for nominal, variation in subshift_inst.x.column_aliases.items():
            # only apply aliases of event weights (of this dataset)
            if nominal not in event_weights:
                continue
            nominal_arr = Route(nominal).apply(events).to_numpy()
            modifier = np.ones_like(nominal_arr)
            nonzero = np.nonzero(nominal_arr)
            modifier[nonzero] = Route(variation).apply(events).to_numpy()[nonzero] / nominal_arr[nonzero]
            subshift_weight = subshift_weight * modifier
        all_weights.append(subshift_weight)

    return events, np.transpose(all_weights)


def load_subshift_chunk(self: HistProducer, events: ak.Array, task: law.Task) -> DotDict:
    """
    Reads the entries of the current chunk from the reduced events and the subshifts' ProduceColumns outputs and
    returns, per selection-dependent subshift, the shifted events, their weight and their category ids.
    """
    rows = ak.to_numpy(events[ROW_COLUMN])
    entry_start, entry_stop = int(rows[0]), int(rows[-1]) + 1
    if entry_stop - entry_start != len(events):
        raise ValueError(f"shifts_hist: entries of the chunk are not contiguous ({entry_start} to {entry_stop - 1})")

    # chunk position of CreateHistograms' own reader (all its inputs use the same positions); task.chunked_io is the
    # ChunkedIOHandler that ChunkedIOMixin.iter_chunked_io sets on the task during iteration (framework internal)
    if getattr(task, "chunked_io", None) is None:
        raise RuntimeError(
            "shifts_hist: task.chunked_io is not set; it needs the ChunkedIOHandler that "
            "ChunkedIOMixin.iter_chunked_io sets on the task while iterating",
        )
    chunk_pos = [pos for pos in task.chunked_io.chunk_positions[0] if pos.entry_start == entry_start]
    if len(chunk_pos) != 1 or chunk_pos[0].entry_stop != entry_stop:
        raise ValueError(f"shifts_hist: no chunk position for entries {entry_start} to {entry_stop - 1}")
    chunk_pos = chunk_pos[0]

    def read(reader: ChunkedParquetReader) -> ak.Array:
        return reader.materialize(
            chunk_index=chunk_pos.index,
            entry_start=chunk_pos.entry_start,
            entry_stop=chunk_pos.entry_stop,
            max_chunk_size=chunk_pos.max_chunk_size,
        )

    reduced_events = read(self.reduced_reader)
    shifted = {}
    for subshift_inst in self.sel_subshift_insts:
        # same steps as CreateHistograms.run for a --shift <subshift> job
        shifted_events = task.merge_chunk_columns(
            reduced_events,
            [read(reader) for reader in self.subshift_readers[subshift_inst.name]],
            subshift_inst.x("column_aliases", {}),
        )
        shifted[subshift_inst.name] = DotDict(
            events=shifted_events,
            weight=event_weight(self, shifted_events),
            category_ids=task.get_category_ids(shifted_events, self.leaf_category_ids),
        )

    return DotDict(entry_start=entry_start, entry_stop=entry_stop, shifted=shifted)


@shifts_hist.init
def shifts_hist_init(self: HistProducer) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    weight_columns = set()

    # add used weight columns and declare shifts that the produced event weight depends on
    if self.dataset_inst.is_mc:
        if self.config_inst.has_aux("event_weights"):
            weight_columns |= {Route(column) for column in self.config_inst.x.event_weights}
            for shift_insts in self.config_inst.x.event_weights.values():
                self.shifts |= {shift_inst.name for shift_inst in shift_insts}

        # optionally also for weights defined by a dataset
        if self.dataset_inst.has_aux("event_weights"):
            weight_columns |= {Route(column) for column in self.dataset_inst.x("event_weights", [])}
            for shift_insts in self.dataset_inst.x.event_weights.values():
                self.shifts |= {shift_inst.name for shift_inst in shift_insts}
    else:
        if self.config_inst.has_aux("data_event_weights"):
            weight_columns |= {Route(column) for column in self.config_inst.x.data_event_weights}
            for shift_insts in self.config_inst.x.data_event_weights.values():
                self.shifts |= {shift_inst.name for shift_inst in shift_insts}

    # add weight columns to uses
    self.uses |= weight_columns


@shifts_hist.post_init
def shifts_hist_post_init(self: HistProducer, task: law.Task, **kwargs) -> None:
    set_subshift_insts(self, task)

    # weight variations and the alias targets of the selection-dependent subshifts (e.g. Jet.pt_jer_up)
    self.uses |= set(map(optional, {c for s in self.subshift_insts for c in s.x("column_aliases", {}).values()}))
    if self.sel_subshift_insts and is_collection_job(self, task):
        self.uses.add(ROW_COLUMN)


@shifts_hist.create_hist
def shifts_hist_create_hist(
    self: HistProducer,
    variables: list[od.Variable],
    task: law.Task,
    **kwargs,
) -> hist.Histogram:
    """
    Histogram with the variable axes and the integer category, process and shift axes, with weight storage.
    """
    return create_hist_from_variables(
        *variables,
        categorical_axes=(
            ("category", "intcat"),
            ("process", "intcat"),
            ("shift", "intcat"),
        ),
        weight=True,
    )


@shifts_hist.fill_hist
def shifts_hist_fill_hist(
    self: HistProducer,
    h: hist.Hist,
    data: dict[str, Any],
    variables: list[od.Variable],
    events: ak.Array,
    task: law.Task,
) -> None:
    """
    Fill the histogram. On a shift collection, once per weight-only subshift from *data* (nominal arrays, one weight
    column per subshift) and once per selection-dependent subshift from its shifted events of this chunk (variable
    selections, categories, processes and values evaluated on the shifted events with CreateHistograms.get_fill_data).
    """
    variable_insts = [
        self.config_inst.get_variable(key)
        for key in data
        if key not in ("category", "process", "shift", "weight")
    ]
    new_data = clean_fill_data(data, variable_insts)
    _fill = lambda d: fill_hist(h, d, last_edge_inclusive=task.last_edge_inclusive)

    if not is_collection_job(self, task):
        _fill(new_data)
        return

    for i, subshift_inst in enumerate(self.weight_subshift_insts):
        _fill(new_data | {"shift": subshift_inst.id, "weight": new_data["weight"][:, i]})

    if not self.sel_subshift_insts:
        return

    # check that the shifted events belong to the chunk of the nominal ones
    chunk = self.subshift_chunk
    rows = ak.to_numpy(events[ROW_COLUMN])
    if chunk is None or (len(rows) and (rows.min() < chunk.entry_start or rows.max() >= chunk.entry_stop)):
        raise RuntimeError("shifts_hist: the shifted events of the selection-dependent subshifts are not of this chunk")

    for subshift_inst in self.sel_subshift_insts:
        shifted = chunk.shifted[subshift_inst.name]
        fill_data, _ = task.get_fill_data(
            events=shifted.events,
            weight=shifted.weight,
            category_ids=shifted.category_ids,
            variable_insts=variables,
            shift_id=subshift_inst.id,
        )
        _fill(clean_fill_data(fill_data, variables))


@shifts_hist.post_process_hist
def shifts_hist_post_process_hist(self: HistProducer, h: hist.Histogram, task: law.Task) -> hist.Histogram:
    """
    Post-process the histogram, converting integer to string axis for consistent lookup across configs where ids might
    be different.
    """
    axis_names = {ax.name for ax in h.axes}

    # translate axes
    if "category" in axis_names:
        category_map = {cat.id: cat.name for cat in self.config_inst.get_leaf_categories()}
        h = translate_hist_intcat_to_strcat(h, "category", category_map)
    if "process" in axis_names:
        process_map = {proc_id: self.config_inst.get_process(proc_id).name for proc_id in h.axes["process"]}
        h = translate_hist_intcat_to_strcat(h, "process", process_map)
    if "shift" in axis_names:
        shift_map = {s.id: s.name for s in [task.global_shift_inst, *self.subshift_insts]}
        h = translate_hist_intcat_to_strcat(h, "shift", shift_map)

    return h


@shifts_hist.requires
def shifts_hist_requires(self: HistProducer, task: law.Task, reqs: dict) -> None:
    """
    On a shift collection, requires the ProduceColumns (and MLEvaluation) outputs of every selection-dependent subshift.
    """
    set_subshift_insts(self, task)
    if not (self.sel_subshift_insts and is_collection_job(self, task)):
        return

    reqs["subshift_producers"] = {
        subshift_inst.name: [
            task.reqs.ProduceColumns.req(task, shift=subshift_inst.name, producer=producer_inst.cls_name)
            for producer_inst in task.producer_insts
            if producer_inst.produced_columns
        ]
        for subshift_inst in self.sel_subshift_insts
    }

    if task.ml_model_insts:
        reqs["subshift_ml"] = {
            subshift_inst.name: [
                task.reqs.MLEvaluation.req(task, shift=subshift_inst.name, ml_model=ml_model_inst.cls_name)
                for ml_model_inst in task.ml_model_insts
            ]
            for subshift_inst in self.sel_subshift_insts
        }


@shifts_hist.setup
def shifts_hist_setup(
    self: HistProducer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    """
    On a shift collection with selection-dependent subshifts, opens readers for the reduced events and the subshifts'
    ProduceColumns (and MLEvaluation) outputs (they are not added to *reader_targets*, which CreateHistograms would
    merge into the nominal events), and adds a temporary file with the entry index ``ROW_COLUMN`` to *reader_targets*,
    so that the hist producer knows which entries the current chunk holds.
    """
    self.subshift_chunk = None
    self.subshift_readers = {}
    self.reduced_reader = None
    self.localize_stack = None
    self.row_target = None
    if not (self.sel_subshift_insts and is_collection_job(self, task)):
        return

    # the shifted events are merged, categorized and filled with methods of the (Ghent) CreateHistograms task
    missing_methods = [name for name in TASK_METHODS if not callable(getattr(task, name, None))]
    if missing_methods:
        raise RuntimeError(
            f"shifts_hist: selection-dependent subshifts need the task methods {', '.join(missing_methods)} of "
            f"CreateHistograms, which {task.task_family} does not provide",
        )

    # columns as read by CreateHistograms.run, plus the alias targets of all subshifts (in uses, see post_init)
    read_columns = {Route("process_id")} | set(map(Route, task.category_id_columns)) | set(self.used_columns)
    for variable_inst in map(self.config_inst.get_variable, law.util.flatten(task.variable_tuples.values())):
        if isinstance(variable_inst.expression, str):
            read_columns.add(Route(variable_inst.expression))
        read_columns |= set(map(Route, law.util.make_set(variable_inst.x("inputs", []))))
    open_options = {"columns": sorted(Route(c).string_column for c in read_columns)}

    # the subshifts read the nominal reduced events: a direct --shift <subshift> job reads ProvideReducedEvents at the
    # subshift, which is the same output as long as no calibrator, selector or reducer registers the shift
    def reduced_paths(**kwargs) -> list[str]:
        return sorted(t.path for t in law.util.flatten(task.reqs.ProvideReducedEvents.req(task, **kwargs).output()))
    nominal_paths = reduced_paths()
    for subshift_inst in self.sel_subshift_insts:
        if reduced_paths(shift=subshift_inst.name) != nominal_paths:
            raise ValueError(
                f"shifts_hist: the reduced events of subshift {subshift_inst.name} differ from those of "
                f"{task.local_shift_inst.name} (a calibrator, selector or reducer registers the shift); shifts_hist "
                f"reads one set of reduced events for all subshifts, run {subshift_inst.name} as a normal --shift job",
            )

    self.leaf_category_ids = {cat.id for cat in self.config_inst.get_leaf_categories()}

    # the reduced events (task inputs are localized during CreateHistograms.run)
    self.reduced_reader = ChunkedParquetReader(task.input()["events"]["events"].abspath, open_options)

    # localize the subshift inputs until teardown
    targets = {
        subshift_inst.name: (
            [inp["columns"] for inp in inputs["subshift_producers"][subshift_inst.name]] +
            [inp["mlcolumns"] for inp in inputs.get("subshift_ml", {}).get(subshift_inst.name, [])]
        )
        for subshift_inst in self.sel_subshift_insts
    }
    self.localize_stack = contextlib.ExitStack()
    local_targets = self.localize_stack.enter_context(law.localize_file_targets(targets, mode="r"))
    self.subshift_readers = {
        name: [ChunkedParquetReader(target.abspath, open_options) for target in subshift_targets]
        for name, subshift_targets in local_targets.items()
    }
    for name, readers in self.subshift_readers.items():
        if any(len(reader) != len(self.reduced_reader) for reader in readers):
            raise ValueError(f"shifts_hist: inputs of subshift {name} differ in length from the reduced events")

    # entry index of each event, read chunk-wise and merged into the events by CreateHistograms
    self.row_target = law.LocalFileTarget(is_tmp="parquet")
    ak.to_parquet(ak.Array({ROW_COLUMN: np.arange(len(self.reduced_reader), dtype=np.int64)}), self.row_target.abspath)
    reader_targets[ROW_COLUMN] = self.row_target


@shifts_hist.teardown
def shifts_hist_teardown(self: HistProducer, task: law.Task, **kwargs) -> None:
    readers = [getattr(self, "reduced_reader", None)] + law.util.flatten(getattr(self, "subshift_readers", {}))
    localize_stack = getattr(self, "localize_stack", None)
    row_target = getattr(self, "row_target", None)
    self.subshift_chunk = None
    self.subshift_readers = {}
    self.reduced_reader = None
    self.localize_stack = None
    self.row_target = None
    try:
        for reader in readers:
            if reader is not None:
                reader.close()
    finally:
        try:
            if localize_stack is not None:
                localize_stack.close()
        finally:
            if row_target is not None:
                row_target.remove()
