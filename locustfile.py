from locust import task, run_single_user
from locust import FastHttpUser


class localhost(FastHttpUser):
    host = "http://localhost:8000"
    default_headers = {
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Connection": "keep-alive",
        "Cookie": 'csrftoken=0vprcG9R5GlNUjYjvbpNzXSq3wkk1h9Z; username-localhost-8888="2|1:0|10:1715035346|23:username-localhost-8888|200:eyJ1c2VybmFtZSI6ICIzN2Q5MmJiYTQ0ZWI0NTVkOTNiYjcxNTExNDVjOTc4MiIsICJuYW1lIjogIkFub255bW91cyBFdXJ5ZG9tZSIsICJkaXNwbGF5X25hbWUiOiAiQW5vbnltb3VzIEV1cnlkb21lIiwgImluaXRpYWxzIjogIkFFIiwgImNvbG9yIjogbnVsbH0=|3882bbbad9345b3c0de47864dbdb73e9c3a307102a8a806e04c85c274155d4b9"; _xsrf=2|7c11f5eb|59cbc45183404581819f601172bec5f6|1715035346',
        "Host": "localhost:8000",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "sec-ch-ua": '"Chromium";v="124", "Microsoft Edge";v="124", "Not-A.Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    }

    @task
    def t(self):
        with self.client.request(
            "GET",
            "/",
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Cache-Control": "max-age=0",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            },
            catch_response=True,
        ) as resp:
            pass
        with self.rest(
            "GET",
            "/api/data/sh/visitorsbycountry/stats",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "GET",
            "/api/data/sh/stats",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "GET",
            "/api/data/sh/visitors/sum/y/2024/01/01",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "POST",
            "/api/data/sh/visitors/raw/m/2011-1-1/2024-5-3",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Content-Length": "0",
                "Origin": "http://localhost:8000",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "POST",
            "/api/data/sh/hotel/raw/m/2011-1-1/2024-5-3",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Content-Length": "0",
                "Origin": "http://localhost:8000",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "GET",
            "/api/data/sh/visitors/sum/m/2024/5/01",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "GET",
            "/api/data/sh/hotel/yoy/m/2024/5/01",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "GET",
            "/api/data/sh/visitors/yoy/y/2024/01/01",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass
        with self.rest(
            "GET",
            "/api/data/sh/visitors/yoy/m/2024/5/01",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "http://localhost:8000/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
            },
        ) as resp:
            pass


if __name__ == "__main__":
    run_single_user(localhost)
