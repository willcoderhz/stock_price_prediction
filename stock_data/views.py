
import requests
from django.http import JsonResponse, HttpResponse
from .models import StockPrice
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
import tempfile
from django.db.models import Q
import os
from dotenv import load_dotenv

# load .env document
load_dotenv()

# 从环境变量中获取 API 密钥
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# 确保 API_KEY 存在
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables.")

def fetch_stock_data(request, symbol='MSFT'):  # 默认符号改为微软 (MSFT)
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full'
    response = requests.get(url)

    if response.status_code != 200:
        return JsonResponse({'error': 'Failed to fetch data from Alpha Vantage'}, status=500)

    data = response.json().get('Time Series (Daily)', {})

    if not data:
        return JsonResponse({'error': 'No data found'}, status=404)

    # 只保留两年内的数据
    two_years_ago = datetime.now().date() - timedelta(days=2*365)

    for date_str, values in data.items():
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # 过滤出两年内的数据
        if date < two_years_ago:
            continue

        open_price = values['1. open']
        high_price = values['2. high']
        low_price = values['3. low']
        close_price = values['4. close']
        volume = values['5. volume']

        # 创建或更新数据库中的股票记录
        StockPrice.objects.update_or_create(
            symbol=symbol,
            date=date,
            defaults={
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'close_price': close_price,
                'volume': volume
            }
        )

    return JsonResponse({'message': 'Data fetched and stored successfully'})




def backtest_strategy(request, symbol='MSFT'):
    # 获取用户输入参数，默认初始金额为10000，短期均线20天，长期均线50天
    initial_investment = float(request.GET.get('initial_investment', 10000))
    short_window = int(request.GET.get('short_window', 5))
    long_window = int(request.GET.get('long_window', 30))

    # 获取存储的股票历史数据（按日期升序排序）
    stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')

    if not stock_data.exists():
        return JsonResponse({'error': 'No data found for this symbol'}, status=404)

    # 准备股票收盘价数据
    prices = np.array([float(item.close_price) for item in stock_data])

    # 计算短期均线和长期均线
    short_mavg = np.convolve(prices, np.ones(short_window), 'valid') / short_window
    long_mavg = np.convolve(prices, np.ones(long_window), 'valid') / long_window

    # 确保移动平均数组的长度与价格数组一致
    min_length = min(len(short_mavg), len(long_mavg))

    # 截取到相同长度的部分
    short_mavg = short_mavg[:min_length]
    long_mavg = long_mavg[:min_length]
    prices = prices[-min_length:]  # 从后面截取相同长度的价格数据

    # 设置交易参数
    holding = False  # 当前是否持有股票
    portfolio_value = initial_investment  # 初始投资金额
    shares = 0  # 当前持有股票的数量
    num_trades = 0  # 交易次数
    max_drawdown = 0  # 最大回撤
    peak_portfolio_value = initial_investment  # 投资组合的峰值

    # 模拟买卖操作
    for i in range(min_length):
        # 确保索引不超出 prices 数组的长度
        if i + long_window - 1 >= len(prices):
            break

        # 如果短期均线高于长期均线且未持有股票，买入
        if short_mavg[i] > long_mavg[i] and not holding:
            shares = portfolio_value / prices[i + long_window - 1]
            portfolio_value = 0  # 用于全仓买入股票
            holding = True
            num_trades += 1
        # 如果短期均线低于长期均线且持有股票，卖出
        elif short_mavg[i] < long_mavg[i] and holding:
            portfolio_value = shares * prices[i + long_window - 1]
            shares = 0
            holding = False
            num_trades += 1

        # 更新组合峰值
        if portfolio_value > peak_portfolio_value:
            peak_portfolio_value = portfolio_value

        # 计算最大回撤
        current_drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

    # 如果最后一天仍然持有股票，按最后一天的价格卖出
    if holding:
        portfolio_value = shares * prices[-1]

    # 计算总回报
    total_return = (portfolio_value - initial_investment) / initial_investment * 100

    # 输出结果
    return JsonResponse({
        'total_return': total_return,
        'num_trades': num_trades,
        'max_drawdown': max_drawdown * 100,  # 以百分比形式显示
        'final_portfolio_value': portfolio_value,
        'message': 'Backtesting completed successfully'
    })


def predict_stock_price(request, symbol='MSFT'):
    # 加载预训练模型
    try:
        with open('ml_models/stock_price_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return JsonResponse({'error': 'Model file not found'}, status=500)

    # 删除该股票符号的已有预测数据
    StockPrice.objects.filter(symbol=symbol).update(predicted_close_price=None)

    # 获取存储的股票历史数据（按日期升序排序）
    stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')

    if not stock_data.exists():
        return JsonResponse({'error': 'No data found for this symbol'}, status=404)

    # 准备股票收盘价数据用于预测，并过滤掉 None 值
    prices = np.array([float(item.close_price) for item in stock_data if item.close_price is not None])  # 转换为 float

    # 检查收盘价是否有有效数据
    if len(prices) == 0:
        return JsonResponse({'error': 'No valid closing prices found for this symbol'}, status=500)

    last_day = len(prices)  # 获取最后一天的天数索引
    last_close_price = prices[-1]  # 获取最后一个有效的收盘价

    # 预测未来30天的股票价格（此处基于线性模型的趋势）
    future_days = np.array([[last_day + i] for i in range(1, 31)])  # 未来30天
    predicted_price_deltas = model.predict(future_days)

    # 将预测值基于最后一个实际收盘价进行调整
    predicted_prices = last_close_price + (predicted_price_deltas - predicted_price_deltas[0])

    # 获取最后一天的日期，并生成未来30天的日期
    last_date = stock_data.last().date
    if last_date is None:
        return JsonResponse({'error': 'Last date is not available in the stock data'}, status=500)

    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

    # 将预测的股票价格保存到数据库中
    for date, predicted_price in zip(future_dates, predicted_prices):
        StockPrice.objects.update_or_create(
            symbol=symbol,
            date=date,
            defaults={'predicted_close_price': float(predicted_price)}  # 保存为 float 类型
        )

    # 将预测结果返回为 JSON
    return JsonResponse({
        'predicted_prices': predicted_prices.tolist(),
        'message': 'Stock price prediction completed and saved successfully'
    })



def generate_report(request, symbol='MSFT'):
    # Fetch historical and predicted stock data
    stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')
    actual_prices = [float(item.close_price) for item in stock_data if item.close_price is not None]  # Convert to float
    predicted_prices = [float(item.predicted_close_price) for item in stock_data if item.predicted_close_price is not None]  # Convert to float
    dates = [item.date for item in stock_data if item.close_price is not None]
    
    if not actual_prices or not predicted_prices:
        return JsonResponse({'error': 'No sufficient data for report generation'}, status=404)
    
    # Calculate financial metrics
    total_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100
    num_trades = len(predicted_prices)

    # Calculate maximum drawdown
    running_max = np.maximum.accumulate(actual_prices)
    drawdowns = (running_max - actual_prices) / running_max * 100
    max_drawdown = np.max(drawdowns)

    # Calculate Sharpe Ratio (assuming risk-free rate is 0)
    returns = np.diff(actual_prices) / actual_prices[:-1]  # Daily return calculation
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe Ratio

    # Generate comparison plot (start predicted prices after the actual prices end)
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')

    # Plot future predicted prices after the last actual date
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, len(predicted_prices)+1)]
    plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')

    # Enhance plot aesthetics: Add date, currency symbol, and title
    plt.title(f'Stock Price Comparison for {symbol} (USD)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()

    # Save the plot as an image
    temp_image_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_image_file.name, format='png')
    temp_image_file.seek(0)

    # Generate the PDF report
    def generate_pdf():
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)

        # Add title with formatting
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(100, 750, f"Stock Performance Report for {symbol}")
        
        # Add date range and metrics
        pdf.setFont("Helvetica", 12)
        pdf.drawString(100, 730, f"Date Range: {dates[0]} - {dates[-1]}")
        pdf.drawString(100, 710, f"Total Return: {total_return:.2f}%")
        pdf.drawString(100, 690, f"Number of Trades: {num_trades}")
        pdf.drawString(100, 670, f"Maximum Drawdown: {max_drawdown:.2f}%")
        pdf.drawString(100, 650, f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Add stock price comparison image
        pdf.drawImage(temp_image_file.name, 100, 400, width=400, height=200)
        
        # Add footer
        pdf.setFont("Helvetica-Oblique", 10)
        pdf.drawString(100, 380, "Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        return buffer

    # Return the PDF or JSON response based on user request
    if request.GET.get('format') == 'pdf':
        pdf_buffer = generate_pdf()
        response = HttpResponse(pdf_buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{symbol}_report.pdf"'
        return response
    else:
        # Convert the image to base64 for JSON response
        image_base64 = base64.b64encode(temp_image_file.read()).decode('utf-8')
        return JsonResponse({
            'total_return': total_return,
            'num_trades': num_trades,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'image_base64': image_base64,
            'date_range': f"{dates[0]} - {dates[-1]}",
            'message': 'Report generated successfully'
        })