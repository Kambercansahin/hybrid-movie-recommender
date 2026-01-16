# Basit demo login
users = {
    "1": {"username": "can", "password": "123"},
    "2": {"username": "demo", "password": "123"},
}

def authenticate(username: str, password: str):
    for uid, u in users.items():
        if u["username"] == username and u["password"] == password:
            return uid
    return None

user_ratings = {
    "1": {},
    "2": {},
}
