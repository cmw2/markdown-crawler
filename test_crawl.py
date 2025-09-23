from web_to_markdown import WebCrawler, MarkdownConverter

# Initialize crawler
crawler = WebCrawler(base_url="https://example.com/docs", max_depth=3)

# Crawl pages
pages = crawler.crawl()

# Convert to markdown
converter = MarkdownConverter()
markdown_content = converter.convert_to_markdown(pages)

# Save the result
converter.save_markdown(markdown_content, "output.md")