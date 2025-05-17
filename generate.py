from random import randrange, choice, choices, shuffle
import datetime, csv


def get_card(pay_system, bank):
    start_card = str(pay_systems[pay_system][bank])
    end_card = ''.join([str(randrange(0, 10)) for _ in range(10)])
    return start_card + end_card


def get_random_person():
    sex = randrange(0, 2)
    if sex:
        name = choice(men_names).strip("\n")
        surname = choice(men_surnames).strip("\n")
        patr = choice(men_patr).strip("\n")
    else:
        name = choice(women_names).strip("\n")
        surname = choice(women_surnames).strip("\n")
        patr = choice(women_patr).strip("\n")
    passport = ''.join([str(randrange(0, 10)) for _ in range(10)])
    while passport in passports or passport[0] == "0":
        passport = ''.join([str(randrange(0, 10)) for _ in range(10)])
    passports.append(passport)
    return (surname + ' ' + name + ' ' + patr), passport


def get_random_datetime():
    date_from = choice(date)
    datetime_from = date_from

    local_date = []
    step_hour = datetime.timedelta(hours=1)
    date_from += datetime.timedelta(days=1)
    for _ in range(1, 150):
        date_from += step_hour
        local_date.append(date_from)
    date_to = choice(local_date)

    return datetime_from, date_to


file_men_names = open("data/men.txt", encoding="utf8")
men_names = file_men_names.readlines()
file_men_names.close()
file_women_names = open("data/women.txt", encoding="utf8")
women_names = file_women_names.readlines()
file_women_names.close()
file_men_surnames = open("data/men_surnames.txt", encoding="utf8")
men_surnames = file_men_surnames.readlines()
file_men_surnames.close()
file_women_surnames = open("data/women_surnames.txt", encoding="utf8")
women_surnames = file_women_surnames.readlines()
file_women_surnames.close()
file_men_patr = open("data/men_patronymics.txt", encoding="utf8")
men_patr = file_men_patr.readlines()
file_men_patr.close()
file_women_patr = open("data/women_patronymics.txt", encoding="utf8")
women_patr = file_women_patr.readlines()
file_women_patr.close()

dataset_size = int(input("Введите кол-во строк для генерации: "))
payments_chance = input(
    "Введите процентное соотношение платежных систем Visa / Mir Pay / MasterCard\nчерез '/'. Пример - 10/60/30:    ")
banks_chance = input(
    "Введите процентное соотношение банков СберБанк / T-Банк / Альфа-Банк\nчерез '/'. Пример - 10/60/30:    ")
payments_chance_list = list(map(float, payments_chance.split("/")))
banks_chance_list = list(map(float, banks_chance.split("/")))

cards = {}
payments_chance = {}
p_check = 0
b_check = 0
banks_chance = {}
pay_systems = {"Visa": {"СберБанк": 427966, "Т-Банк": 437773, "Альфа-Банк": 410584},
               "Mir Pay": {"СберБанк": 227901, "Т-Банк": 221624, "Альфа-Банк": 219539},
               "MasterCard": {"СберБанк": 546935, "Т-Банк": 521324, "Альфа-Банк": 515429}}
for i, key in enumerate(pay_systems.keys()):
    payments_chance[key] = payments_chance_list[i] / 100
for i, key in enumerate(pay_systems["Visa"].keys()):
    banks_chance[key] = banks_chance_list[i] / 100
passports = []
cities_file = open("data/cities.txt", mode="r", encoding="utf8")
cities = cities_file.readlines()
start_date = '2023-05-03'
finish_date = '2024-05-03'
date = []
start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end = datetime.datetime.strptime(finish_date, '%Y-%m-%d')
step = datetime.timedelta(hours=1)
while start <= end:
    start += step
    date.append(start)

seats_reserved = []
cities_flights = []
time_flights = []
nums_flights = [i for i in range(1, 299)] + [i for i in range(301, 599)] + [i for i in range(701, 799)]
random_nums_flights = nums_flights.copy()
shuffle(random_nums_flights)
for _ in range(len(nums_flights)):
    seats_reserved.append([])
    time_1, time_2 = get_random_datetime()
    time_flights.append([time_1, time_2])

    city_from = choice(cities)
    city_to = choice(cities)
    while city_to == city_from:
        city_to = choice(cities)
    cities_flights.append([city_from.strip(), city_to.strip()])

all_card_variants = []
weights = []
for payment in pay_systems.keys():
    for bank in pay_systems[payment].keys():
        all_card_variants.append([payment, bank])
        weights.append(payments_chance[payment] * banks_chance[bank])
file = open("dataset20k.csv", mode="w", encoding="utf8")
file_writer = csv.writer(file, delimiter=";")
file_writer.writerow("Name;Passport;City from;City to;Date from;Date to;Train;Seat;Cost;Card".split(";"))
if dataset_size <= 680000:
    for i in range(dataset_size):
        person = []
        person.extend(list(get_random_person()))
        while True:

            random_flight = choice(random_nums_flights[:dataset_size // 1000 + 14])
            index_flight = nums_flights.index(random_flight)
            seat = str(randrange(1, 11)) + "-" + str(randrange(1, 101))
            if seat not in seats_reserved[index_flight]:
                seats_reserved[index_flight].append(seat)
                break
        person.extend(cities_flights[index_flight])
        person.extend(time_flights[index_flight])
        person.append(random_flight)
        person.append(seat)
        if 1 <= random_flight <= 150 or 301 <= random_flight <= 450:
            person.append(str(randrange(2000, 5000)))
        elif 151 <= random_flight <= 298 or 451 <= random_flight <= 598:
            person.append(str(randrange(3000, 6500)))
        elif 701 <= random_flight <= 750:
            person.append(str(randrange(4000, 15000)))
        else:
            person.append(str(randrange(5000, 30000)))
        cards_data = choices(all_card_variants, weights)[0]
        random_card = get_card(cards_data[0], cards_data[1])
        while random_card in cards and cards[random_card] >= 5:
            cards_data = choices(all_card_variants, weights)
            random_card = get_card(cards_data[0], cards_data[1])
        if random_card not in cards:
            cards[random_card] = 1
        elif cards[random_card] < 5:
            cards[random_card] += 1
        person.append(random_card)
        file_writer.writerow(person)
else:
    for i in range(dataset_size):
        print(i)
        person = []
        person.extend(list(get_random_person()))
        while True:
            random_flight = choice(nums_flights)
            index_flight = nums_flights.index(random_flight)
            seat = str(randrange(1, (dataset_size // 50000))) + "-" + str(randrange(1, 101))
            if seat not in seats_reserved[index_flight]:
                seats_reserved[index_flight].append(seat)
                break
        person.extend(cities_flights[index_flight])
        person.extend(time_flights[index_flight])
        person.append(random_flight)
        person.append(seat)
        if 1 <= random_flight <= 150 or 301 <= random_flight <= 450:
            person.append(str(randrange(2000, 5000)))
        elif 151 <= random_flight <= 298 or 451 <= random_flight <= 598:
            person.append(str(randrange(3000, 6500)))
        elif 701 <= random_flight <= 750:
            person.append(str(randrange(4000, 15000)))
        else:
            person.append(str(randrange(5000, 30000)))
        cards_data = choices(all_card_variants, weights)[0]
        random_card = get_card(cards_data[0], cards_data[1])
        while random_card in cards and cards[random_card] >= 5:
            cards_data = choices(all_card_variants, weights)
            random_card = get_card(cards_data[0], cards_data[1])
        if random_card not in cards:
            cards[random_card] = 1
        elif cards[random_card] < 5:
            cards[random_card] += 1
        person.append(random_card)
        file_writer.writerow(person)

file.close()
print("Сгенерирован строк:", dataset_size)