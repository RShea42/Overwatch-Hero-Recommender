import requests

HERO_URL = "https://overfast-api.tekrop.fr/heroes/soldier-76"


def main():
    try:
        response = requests.get(HERO_URL, timeout=10)
        response.raise_for_status()
        hero = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving hero data: {e}")
        return

    print(f"Name: {hero.get('name')}")
    print(f"Role: {hero.get('role')}")

    health = hero.get("hitpoints", {})
    print("Health:")
    print(f"  Health: {health.get('health')}")
    print(f"  Armor: {health.get('armor')}")
    print(f"  Shields: {health.get('shields')}")

    print("Abilities:")
    for ability in hero.get("abilities", []):
        print(f"  {ability.get('name')}: {ability.get('description')}")


if __name__ == "__main__":
    main()
