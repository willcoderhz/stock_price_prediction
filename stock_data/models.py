from django.db import models

class StockPrice(models.Model):
    symbol = models.CharField(max_length=10)  # 股票符号
    date = models.DateField()  # 日期
    open_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # 开盘价
    close_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # 收盘价
    high_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # 最高价
    low_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # 最低价
    volume = models.BigIntegerField(null=True)  # 成交量
    predicted_close_price = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # 预测收盘价

    def __str__(self):
        return f"{self.symbol} - {self.date}"
