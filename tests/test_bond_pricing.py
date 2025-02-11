import unittest
from models.bond_pricing import BondPricing

class TestBondPricing(unittest.TestCase):

    def setUp(self):
        self.bond_pricing = BondPricing()

    def test_calculate_price(self):
        price = self.bond_pricing.calculate_price(face_value=1000, coupon_rate=0.05, years_to_maturity=10, market_rate=0.04)
        self.assertAlmostEqual(price, 1170.30, places=2)

    def test_calculate_duration(self):
        duration = self.bond_pricing.calculate_duration(face_value=1000, coupon_rate=0.05, years_to_maturity=10, market_rate=0.04)
        self.assertAlmostEqual(duration, 8.11, places=2)

    def test_calculate_convexity(self):
        convexity = self.bond_pricing.calculate_convexity(face_value=1000, coupon_rate=0.05, years_to_maturity=10, market_rate=0.04)
        self.assertAlmostEqual(convexity, 66.67, places=2)

if __name__ == '__main__':
    unittest.main()