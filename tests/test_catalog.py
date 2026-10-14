from backend import main


def sample_movie(movie_id=42, title="Sample Film", language="hi"):
    return {
        "id": movie_id,
        "title": title,
        "original_title": title,
        "overview": "A sample synopsis.",
        "release_date": "2024-01-01",
        "vote_average": 7.8,
        "vote_count": 250,
        "poster_path": "/poster.jpg",
        "backdrop_path": "/backdrop.jpg",
        "original_language": language,
        "genre_ids": [18],
    }


def test_indian_catalog_applies_country_and_language(monkeypatch):
    captured = {}

    def fake_tmdb(path, params):
        captured.update(path=path, params=params)
        return {"page": 1, "total_pages": 4, "total_results": 80, "results": [sample_movie()]}

    monkeypatch.setattr(main, "tmdb_get", fake_tmdb)
    result = main.indian_catalog(page=1, language="hi", sort="popularity.desc")

    assert captured["path"] == "/discover/movie"
    assert captured["params"]["with_origin_country"] == "IN"
    assert captured["params"]["with_original_language"] == "hi"
    assert result["results"][0]["poster"].endswith("/poster.jpg")


def test_hollywood_catalog_uses_us_origin(monkeypatch):
    captured = {}

    def fake_tmdb(path, params):
        captured.update(path=path, params=params)
        return {"page": 1, "total_pages": 2, "total_results": 30, "results": [sample_movie(language="en")]}

    monkeypatch.setattr(main, "tmdb_get", fake_tmdb)
    result = main.hollywood_catalog(page=1, sort="popularity.desc")

    assert captured["params"]["with_origin_country"] == "US"
    assert result["total_results"] == 30


def test_indian_search_excludes_english_language_results(monkeypatch):
    def fake_tmdb(path, params):
        assert path == "/search/movie"
        return {
            "page": 1,
            "total_pages": 1,
            "total_results": 3,
            "results": [
                sample_movie(1, "Hindi Film", "hi"),
                sample_movie(2, "Tamil Film", "ta"),
                sample_movie(3, "English Film", "en"),
            ],
        }

    monkeypatch.setattr(main, "tmdb_get", fake_tmdb)
    result = main.catalog_search(q="film", page=1, indian_only=True)

    assert [movie["title"] for movie in result["results"]] == ["Hindi Film", "Tamil Film"]
    assert result["total_results"] == 2


def test_details_and_recommendations_normalize_tmdb_data(monkeypatch):
    def fake_tmdb(path, params):
        if path == "/movie/42":
            return {
                **sample_movie(),
                "runtime": 120,
                "status": "Released",
                "genres": [{"id": 18, "name": "Drama"}],
                "production_countries": [{"iso_3166_1": "IN", "name": "India"}],
            }
        if path == "/movie/42/recommendations":
            return {"page": 1, "total_pages": 1, "total_results": 1, "results": [sample_movie(99, "Related Film")]}
        raise AssertionError(f"Unexpected TMDB path: {path}")

    monkeypatch.setattr(main, "tmdb_get", fake_tmdb)
    detail = main.catalog_movie_details(42)
    related = main.catalog_movie_recommendations(42)

    assert detail["runtime"] == 120
    assert detail["genres"] == ["Drama"]
    assert detail["production_countries"] == ["India"]
    assert related["results"][0]["id"] == 99
