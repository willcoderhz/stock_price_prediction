from django.db import models

class StockPrice(models.Model):
    symbol = models.CharField(max_length=10)  # Stock symbol
    date = models.DateField()  # Date
    open_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Opening price
    close_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Closing price
    high_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Highest price
    low_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Lowest price
    volume = models.BigIntegerField(null=True)  # Trading volume
    predicted_close_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Predicted closing price

    def __str__(self):
        return f"{self.symbol} - {self.date}"