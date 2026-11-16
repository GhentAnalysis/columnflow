# coding: utf-8

"""
Tests of the shift collection helpers in columnflow.histogramming.cmsGhent.shifts (change Ghent).
"""

__all__ = ["ShiftCollectionsTests"]

import unittest

import order as od

from columnflow.util import DotDict
from columnflow.histogramming.cmsGhent.shifts import add_shift_collection, resolve_shift_collections


class ShiftCollectionsTests(unittest.TestCase):

    def make_config(self) -> od.Config:
        config = od.Config(name="test_config", id=1)
        config.x.event_weights = DotDict({"normalization_weight": []})
        config.add_shift(name="nominal", id=0)
        shift_id = 1
        for source in ["jer", "muon_scale", "weight_a", "weight_b", "pdf_1", "lumi"]:
            for direction in ["up", "down"]:
                config.add_shift(name=f"{source}_{direction}", id=(shift_id := shift_id + 1), type="shape")
        config.get_shift("pdf_1_down").add_tag("mirrored")
        add_shift_collection(config, "sel", 100, ["jer_*", "muon_scale_*"], weight_column="normalization_weight")
        add_shift_collection(config, "base", 200, ["weight_*", "pdf_1_up"])
        return config

    def test_add_shift_collection(self):
        config = self.make_config()
        self.assertEqual(config.get_shift("sel_up").id, 101)
        self.assertEqual(config.get_shift("sel_down").id, 102)
        self.assertEqual(config.get_shift("sel_up").x.subshifts, ["jer_up", "muon_scale_up"])
        self.assertEqual(config.get_shift("base_down").x.subshifts, ["weight_a_down", "weight_b_down"])
        self.assertTrue(config.get_shift("base_up").has_tag("is_collection"))
        self.assertEqual([s.name for s in config.x.event_weights["normalization_weight"]], ["sel_up", "sel_down"])

        # existing collections are skipped
        self.assertEqual(add_shift_collection(config, "sel", 300, ["jer_*"]), [])

    def test_resolve(self):
        config = self.make_config()
        shifts = [
            "nominal", "jer_up", "jer_down", "weight_a_up", "weight_a_down", "pdf_1_up", "pdf_1_down", "lumi_up",
            "lumi_down",
        ]

        # no collections: everything individual
        collection_shifts, individual = resolve_shift_collections(config, shifts, [])
        self.assertEqual(collection_shifts, {})
        self.assertEqual(individual, sorted(set(shifts) - {"nominal"}))

        # one collection, both directions
        collection_shifts, individual = resolve_shift_collections(config, shifts, ["sel"])
        self.assertEqual(collection_shifts, {"sel_up": ["jer_up"], "sel_down": ["jer_down"]})
        self.assertEqual(individual, ["lumi_down", "lumi_up", "pdf_1_down", "pdf_1_up", "weight_a_down", "weight_a_up"])

        # two collections; the mirrored pdf_1_down is in no collection and stays individual
        collection_shifts, individual = resolve_shift_collections(config, shifts, ["sel", "base"])
        self.assertEqual(collection_shifts, {
            "sel_up": ["jer_up"],
            "sel_down": ["jer_down"],
            "base_up": ["pdf_1_up", "weight_a_up"],
            "base_down": ["weight_a_down"],
        })
        self.assertEqual(individual, ["lumi_down", "lumi_up", "pdf_1_down"])

        # a single direction
        collection_shifts, individual = resolve_shift_collections(config, shifts, ["sel_up"])
        self.assertEqual(collection_shifts, {"sel_up": ["jer_up"]})
        self.assertIn("jer_down", individual)

        # collections that contain none of the shifts are dropped
        collection_shifts, _ = resolve_shift_collections(config, ["nominal", "lumi_up", "lumi_down"], ["sel", "base"])
        self.assertEqual(collection_shifts, {})

        # a mirrored subshift stays individual (filled by no task, skipped there)
        add_shift_collection(config, "mirrored_only", 500, ["pdf_1_down"])
        collection_shifts, individual = resolve_shift_collections(config, ["pdf_1_down"], ["mirrored_only"])
        self.assertEqual(collection_shifts, {})
        self.assertEqual(individual, ["pdf_1_down"])

    def test_resolve_errors(self):
        config = self.make_config()

        # unknown collection
        with self.assertRaises(ValueError):
            resolve_shift_collections(config, ["jer_up"], ["no_such_collection"])

        # a shift that is not a collection
        with self.assertRaises(ValueError):
            resolve_shift_collections(config, ["jer_up"], ["jer"])

        # a shift in two requested collections
        add_shift_collection(config, "jer_only", 400, ["jer_*"])
        with self.assertRaisesRegex(ValueError, "jer_up"):
            resolve_shift_collections(config, ["jer_up"], ["sel", "jer_only"])

        # the duplicate check does not depend on the requested shifts
        with self.assertRaisesRegex(ValueError, "more than one"):
            resolve_shift_collections(config, ["lumi_up"], ["sel", "jer_only"])
