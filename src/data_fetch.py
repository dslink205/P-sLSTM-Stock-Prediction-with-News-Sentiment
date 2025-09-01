import json
import datetime
import requests
from polygon import RESTClient
from polygon.rest.models import TickerNews

# 初始化 Polygon 客户端
client = RESTClient("JVrVjquEEsXgZuRaL3hQhzxBSpMSKqp4")

# M7 公司股票代码
tickers = ["GOOGL", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA"]

for ticker in tickers:
    # 用于存储新闻的列表
    news = []

    # 分页拉取所有指定公司的新闻数据
    params = {
        "ticker": ticker,
        "order": "asc",         # 按发布时间升序，从最早开始
        "limit": 1000,          # 每次请求最大 1000 条（付费账户支持）
        "sort": "published_utc"
    }

    while True:
        try:
            response = client.list_ticker_news(**params)
            page_news = []
            for n in response:
                if isinstance(n, TickerNews):
                    page_news.append(n)
            news.extend(page_news)
            print(f"{ticker}: 已拉取 {len(news)} 条新闻...")

            # 检查是否有下一页
            if hasattr(response, 'next_url') and response.next_url:
                params['cursor'] = response.next_url.split('cursor=')[1]
            else:
                break  # 没有下一页，退出循环
        except Exception as e:
            print(f"{ticker}: API 请求错误: {e}")
            break

    # 转换为 JSON 兼容的字典格式
    news_json = []
    for item in news:
        news_item = {
            "id": item.id,
            "title": item.title,
            "publisher": {
                "name": item.publisher.name if item.publisher else "",
                "homepage_url": item.publisher.homepage_url if item.publisher else "",
                "logo_url": item.publisher.logo_url if item.publisher else "",
                "favicon_url": item.publisher.favicon_url if item.publisher else ""
            },
            "published_utc": item.published_utc,
            "article_url": item.article_url,
            "tickers": item.tickers,
            "description": item.description or "",
            "keywords": item.keywords or [],
            "image_url": item.image_url or "",
            "insights": [
                {
                    "ticker": insight.ticker,
                    "sentiment": insight.sentiment,
                    "sentiment_reasoning": insight.sentiment_reasoning
                } for insight in item.insights
            ] if item.insights else [],
            "amp_url": item.amp_url or "",
            "author": item.author or ""
        }
        news_json.append(news_item)

    # 保存为 JSON 文件
    output_path = f"data/raw/{ticker.lower()}_news_polygon.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": news_json}, f, indent=4, ensure_ascii=False)

    # 打印统计信息
    print(f"\n{ticker}: 共获取 {len(news)} 条新闻")
    if news:
        print(f"{ticker}: 最早新闻时间: {min(n.published_utc for n in news)}")
        print(f"{ticker}: 最晚新闻时间: {max(n.published_utc for n in news)}")
        # 打印前 5 条新闻的日期和标题
        print(f"\n{ticker}: 前 5 条新闻：")
        for item in news[:5]:
            print(f"{item.published_utc:<25} {item.title:<50}")
