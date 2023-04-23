import data_collection.bbc_transfer_rumours_scraper as bbc_scraper
import data_collection.transfer_news_scraper as transfer_news_scraper
import data_collection.transfermarkt_scraper as  transfermarkt_scraper
import data_collection.football_api as football_api

def collect_data():
    bbc_scraper.run()
    transfer_news_scraper.run()
    transfermarkt_scraper.run()
    football_api.run()


def main():
    collect_data()
    
if __name__ == "__main__":
    main()
