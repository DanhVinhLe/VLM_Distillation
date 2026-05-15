import unittest
from types import SimpleNamespace


class KDPlumbingTests(unittest.TestCase):
    def test_default_empty_kd_loss_type_uses_ce_only_criterion(self):
        from src.criterions import CEOnlyCriterion, build_criterion

        criterion = build_criterion(SimpleNamespace(kd_loss_type=""))

        self.assertIsInstance(criterion, CEOnlyCriterion)

    def test_emkd_reads_weight_arguments(self):
        from src.criterions.em_kd import EMKDCriterion

        args = SimpleNamespace(
            em_kd_alpha=0.7,
            em_kd_beta=0.11,
            em_kd_gamma=3.5,
            em_kd_temperature=2.25,
        )

        criterion = EMKDCriterion(args)

        self.assertEqual(criterion.alpha, 0.7)
        self.assertEqual(criterion.beta, 0.11)
        self.assertEqual(criterion.gamma, 3.5)
        self.assertEqual(criterion.temperature, 2.25)

    def test_joint_alias_builds_unit_aligned_criterion(self):
        from src.criterions import UnitAlignedDistillationCriterion, build_criterion

        for alias in ("joint", "unit_aligned", "unit_aligned_distillation"):
            with self.subTest(alias=alias):
                criterion = build_criterion(SimpleNamespace(kd_loss_type=alias))
                self.assertIsInstance(criterion, UnitAlignedDistillationCriterion)

    def test_joint_mode_enables_sre_pooler_in_collator(self):
        from src.data.dataset import VlmDistillDataCollator

        collator = VlmDistillDataCollator(
            student_processor=object(),
            teacher_processor=object(),
            data_args=SimpleNamespace(kd_loss_type="joint"),
            model_args=SimpleNamespace(teacher_model_name="teacher"),
        )

        self.assertTrue(collator.use_sre_pooler)


if __name__ == "__main__":
    unittest.main()
