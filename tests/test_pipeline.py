import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.data.preprocess import validate_frame, audit, feature_groups
from src.data.make_dataset import prepare
from src.models.model_config import model_configs
from src.models.predict import predict
from src.utils.utilities import FEATURES, LABELS, ROOT, SEED

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = validate_frame(pd.read_csv(ROOT / 'data/raw/Maternal Health Risk Data Set.csv'))

    def test_audit_retains_conflicts(self):
        frame = pd.concat([self.frame.iloc[:1]] * 2, ignore_index=True)
        frame.loc[1, 'RiskLevel'] = 'low risk'
        report = audit(frame)
        self.assertEqual(report['conflicting_label_groups'], 1)
        self.assertEqual(report['rows_removed'], 0)

    def test_grouped_holdout_and_cv(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch('src.data.make_dataset.PROCESSED', folder), patch('src.data.make_dataset.RESULTS', folder):
                report = prepare(ROOT / 'data/raw/Maternal Health Risk Data Set.csv')
            train = pd.read_csv(folder / 'train.csv')
            test = pd.read_csv(folder / 'test.csv')
            self.assertEqual(len(train) + len(test), len(self.frame))
            self.assertTrue(0.18 < report['split']['test_fraction'] < 0.22)
            groups = feature_groups(train)
            self.assertFalse(set(groups) & set(feature_groups(test)))
            for fit, val in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, train.RiskLevel, groups):
                self.assertFalse(set(groups.iloc[fit]) & set(groups.iloc[val]))
                self.assertEqual(set(train.iloc[val].RiskLevel), set(LABELS))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            validate_frame(pd.DataFrame({'Age': [20]}), False)
        invalid = self.frame.iloc[:1].copy()
        invalid.loc[invalid.index[0], 'RiskLevel'] = 'unknown'
        with self.assertRaises(ValueError):
            validate_frame(invalid)

    def test_every_model_and_serialized_prediction(self):
        import joblib
        sample = self.frame.groupby('RiskLevel', group_keys=False).head(15)
        with tempfile.TemporaryDirectory() as directory:
            for name, estimator, _ in model_configs(quick=True):
                estimator.fit(sample[FEATURES], sample.RiskLevel)
                path = Path(directory) / f'{name}.joblib'
                joblib.dump(estimator, path)
                query = sample[FEATURES].copy()
                query.loc[query.index[0], 'Age'] = float('nan')
                output = predict(query, path)
                self.assertEqual(len(output), len(sample))
                self.assertTrue(set(output.PredictedRisk).issubset(LABELS))

if __name__ == '__main__':
    unittest.main()
