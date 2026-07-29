import html
import re
import urllib.parse
import urllib.request

from thinair import Thing

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


class RedditBot(Thing):
    """Checks reddit posts for the user. Browses reddit over plain HTTP — nowhere else."""

    def _reddit_url(self, url):
        url = str(url)
        if url.startswith("/"):
            return "https://old.reddit.com" + url
        host = urllib.parse.urlparse(url).hostname or ""
        if host != "reddit.com" and not host.endswith(".reddit.com"):
            raise ValueError("RedditBot browses reddit only")
        return url.replace("://www.reddit.com", "://old.reddit.com")

    def _fetch(self, url):
        request = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    def browse(self, url):
        """Open any reddit page (a path like /r/LocalLLaMA/ or a reddit.com URL).
        Returns the post links on the page and its readable text."""
        page = self._fetch(self._reddit_url(url))
        links = []
        for attrs, text in re.findall(r'<a ([^>]*class="[^"]*\btitle[^"]*"[^>]*)>([^<]+)</a>', page):
            href = re.search(r'href="([^"]+)"', attrs)
            if href:
                links.append({"title": html.unescape(text).strip(), "url": href.group(1)})
        bodies = re.findall(r'<div class="md">(.*?)</div>', page, re.S)
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", b)) for b in bodies)
        return {"links": links[:20], "text": re.sub(r"\s+", " ", text).strip()[:3500]}

    def search(self, query):
        """Search all of reddit; returns a list of {title, url} results."""
        return self.browse("/search?q=" + urllib.parse.quote(str(query)))["links"]


a = RedditBot("a bot that follows news about local LLMs and Apple MLX")

mlx_posts = a.check_posts(
    "Apple MLX only",
    returns={"posts": [{"title": str, "url": str}], "mood": str},
)
print(mlx_posts.posts)          # bare list of dicts — schema-guaranteed
print(mlx_posts.mood)           # bare str — the community mood, as one word
print()
print(mlx_posts.summarize())    # the result is a Thing: chain straight into it
