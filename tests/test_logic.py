import unittest
import pandas as pd
from src.kpi_engine import calculate_kpis

class TestManufacturingKPIs(unittest.TestCase):
    def setUp(self):
        # Create a small dummy dataset for testing
        data = {
            'Date': ['2026-01-15', '2026-01-15'],
            'Machine ID': ['M1', 'M1'],
            'Units Produced': [100, 200],
            'Defective Units': [10, 20],
            'Downtime (minutes)': [30, 30]
        }
        self.df = pd.DataFrame(data)

    def test_summary_calculations(self):
        summary, _ = calculate_kpis(self.df)
        
        # Total Units should be 100 + 200 = 300
        self.assertEqual(summary['Total Units'], 300) [cite: 5, 7]
        
        # Total Defects should be 10 + 20 = 30
        self.assertEqual(summary['Total Defects'], 30) [cite: 5]
        
        # Yield should be (270/300) * 100 = 90.0%
        self.assertEqual(summary['Yield %'], 90.0) [cite: 5]
        
        # Avg Downtime should be (30+30)/2 = 30.0
        self.assertEqual(summary['Avg Downtime (min)'], 30.0) [cite: 5]

if __name__ == '__main__':
    unittest.main()