class BondPricing:
    def __init__(self, face_value, coupon_rate, maturity, market_rate):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.market_rate = market_rate

    def price(self):
        price = 0
        for t in range(1, self.maturity + 1):
            price += self.coupon_payment(t) / (1 + self.market_rate) ** t
        price += self.face_value / (1 + self.market_rate) ** self.maturity
        return price

    def coupon_payment(self, t):
        return self.face_value * self.coupon_rate

    def duration(self):
        duration = 0
        for t in range(1, self.maturity + 1):
            duration += (t * self.coupon_payment(t)) / (1 + self.market_rate) ** t
        duration += (self.maturity * self.face_value) / (1 + self.market_rate) ** self.maturity
        return duration / self.price()

    def convexity(self):
        convexity = 0
        for t in range(1, self.maturity + 1):
            convexity += (t * (t + 1) * self.coupon_payment(t)) / (1 + self.market_rate) ** (t + 2)
        convexity += (self.maturity * (self.maturity + 1) * self.face_value) / (1 + self.market_rate) ** (self.maturity + 2)
        return convexity / self.price()