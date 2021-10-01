import scrapy
from scrapy.crawler import CrawlerProcess
import json

actors = []
films = []
titles = set()


def parse_crew(response):
    cast = response.xpath('.//*/a[@data-testid="title-cast-item__actor"]/text()').extract()
    films.append({
        "url": response.meta["url"],
        "title": response.meta["title"],
        "cast": cast
    })
    titles.add(response.meta["url"])


def create_parse_actor(i):
    def parse_actor(response):
        actors[i]['born'] = response.xpath('.//*[@id="name-born-info"]/time/@datetime').extract_first()
        ts = response.xpath('.//*/div[contains(@class, "filmo-row")]/b/a/text()')[:15].extract()
        actors[i]['movies'] = ts
        hrefs = response.xpath('.//*/div[contains(@class, "filmo-row")]/b/a/@href')[:15].extract()

        for title, href in zip(ts, hrefs):
            parts = href.split('/')
            url = f'https://www.imdb.com/{parts[1]}/{parts[2]}/'

            if url not in titles:
                yield scrapy.Request(url, callback=parse_crew, meta={"url": url, "title": title})

    return parse_actor


class ImdbSpider(scrapy.Spider):
    name = "imdb"
    allowed_domains = ["imdb.com"]
    start_urls = ['https://www.imdb.com/search/name/?gender=male%2Cfemale&ref_=nv_cel_m']

    def parse(self, response):
        parent = scrapy.Selector(text=response.xpath('.//*[@class="lister-list"]').extract_first().replace('<br>', '\n'))
        rows = parent.xpath('.//*[@class="lister-item-content"]')

        for i, row in enumerate(rows):
            href = row.xpath(".//*/a/@href").extract_first().strip()
            bio = row.xpath(".//*/text()").extract()[10:]

            yield scrapy.Request(f'https://www.imdb.com{href}/', callback=create_parse_actor(i))

            actors.append({
                "name": row.xpath(".//*/a/text()").extract_first().strip(),
                "url": f'https://www.imdb.com{href}/',
                "bio": ''.join(bio).strip(),
            })


process = CrawlerProcess()
process.crawl(ImdbSpider)
process.start()

with open('./actors.jl', 'w') as f:
    f.write('\n'.join(list(map(json.dumps, actors))))
    f.close()

with open('./films.jl', 'w') as f:
    f.write('\n'.join(list(map(json.dumps, films))))
    f.close()
