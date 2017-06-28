# -*- coding:utf-8 -*-
__news__ = {
        # /*NES数据源里包括的信息，部分信息的名称、数据格式可能不统一，需要转换*/
        "_id": "",
        "url": "",
        "title": "",
        "content": "",
        "publishTime": "2017-04-19 17:00:00",
        "publisher": "",
        "image": [],
        "video": [],
        "language": "",

        # /*以下是元数据抽取能够补充的信息*/

        # /*分词*/
        "seggedTitle": "",
        "seggedContent": "",
        "stemContent": "",
        "lowerContent": "",

        # /*关键词抽取*/
        "keywords": [{"word": "理财", "score": 0.66}, {"word": "互联网", "score": 0.62}],

        # /*实体识别和链接*/
        "persons": [{"mention": "习大大", "count": 2, "linkedURL": "http://xlore.org/instance/xxxxx"}],
        "locations": [{"mention": "北京", "count": 4, "linkedURL": "http://xlore.org/instance/xxxxx"}],
        "organizations": [{"mention": "中国人民银行", "count": 1, "linkedURL": "http://xlore.org/instance/xxxxx"}],
        # /*保留字段：以后做垂直领域（如科技）分析时可能需要的其他专业的实体，暂时不插入数据库*/
        "entities": [{"mention": "深度学习", "count": 1, "linkedURL": "http://xlore.org/instance/xxxxx"}],

        # /*新闻要素*/
        "when": [{"word": "", "score": 0.66}, {"word": "", "score": 0.62}],
        "where": [{"word": "", "score": 0.66}, {"word": "", "score": 0.62}],
        "who": [{"word": "", "score": 0.66}, {"word": "", "score": 0.62}],

        # /*新闻分类*/
        "category": "",

        # /*入库时间*/
        "crawlTime": "2017-04-19 17:00:00"
    }
__event__ = {
        # /*基本信息*/
        "_id": "",
        "label": "",

        # /*创建、最近更新、停止更新时间*/
        # "2017-04-19 15:00:00"
        "created": "",
        "updated": "",
        "closed": "",

        # /*代表新闻*/
        "keynews": {
            "id": "新闻id",
            "title": "新闻标题",
            "url": "原文url",
            "publishTime": "文章时间",
            "abstract": "内容摘要"
        },

        # /*新闻列表，按相关性降序排列*/
        # articles = [{
        #     "id": "新闻id",
        #     "title": "新闻标题",
        #     "url": "原文url",
        #     "publishTime": "文章时间",
        #     "abstract": "内容摘要",
        #     "score": 0.8
        # }]
        "count": 0,
        "articles": [],

        # /*实体列表，按相关性降序排列*/
        "persons": [{"mention": "习大大", "score": 0.8, "linkedURL": "http://xlore.org/instance/xxxxx"}],
        "locations": [{"mention": "北京", "score": 0.6, "linkedURL": "http://xlore.org/instance/xxxxx"}],
        "organizations": [{"mention": "中国人民银行", "score": 0.68, "linkedURL": "http://xlore.org/instance/xxxxx"}],

        # /*关键词列表，降序排列*/
        "keywords": [{"word": "理财", "score": 0.66}, {"word": "互联网", "score": 0.62}],

        # /*事件要素，降序排列*/
        "when": [{"word": "", "score": 0.66}, {"word": "", "score": 0.62}],
        "where": [{"word": "", "score": 0.66}, {"word": "", "score": 0.62}],
        "who": [{"word": "", "score": 0.66}, {"word": "", "score": 0.62}],

        # /*事件层次关系*/
        "childrens": [],
        "father": -1,

        # /*事件相关性，降序排列*/
        "relatedEvents": [{"id": 0, "label": "xxx", "score": 0.66}]

        # /*实体关系（暂不考虑）*/
    }

def get_news_json():
    return __news__.copy()

def get_event_json():
    return __event__.copy()
