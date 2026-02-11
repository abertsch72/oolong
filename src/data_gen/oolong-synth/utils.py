import datetime
import random


def sample_random_dates(
    n, start_date=datetime.date(2022, 2, 16), end_date=datetime.date(2025, 6, 21)
):
    """
    Sample n random dates from within a date range, with repetition allowed.

    Args:
        start_date (datetime.date): The start date of the range (inclusive)
        end_date (datetime.date): The end date of the range (inclusive)
        n (int): Number of random dates to sample

    Returns:
        list: A list of n random datetime.date objects
    """
    # Calculate the number of days in the range
    days_in_range = (end_date - start_date).days + 1

    # Sample n random dates
    random_dates = []
    formatted_dates = []
    for _ in range(n):
        # Generate a random number of days to add to start_date
        random_days = random.randint(0, days_in_range - 1)
        # Create the random date
        random_date = start_date + datetime.timedelta(days=random_days)
        random_dates.append(random_date)

        formatted_date = random_date.strftime("%b %d, %Y")
        formatted_dates.append(formatted_date)

    return random_dates, formatted_dates


def generate_skewed_user_ids(n, min_id=10000, max_id=99999):
    """
    Generate n user IDs where approximately 80% of the IDs come from
    20% of the most common IDs, following a Pareto-like distribution.

    Args:
        n (int): Number of user IDs to generate
        min_id (int): Minimum user ID (default: 10000 for 5-digit numbers)
        max_id (int): Maximum user ID (default: 99999 for 5-digit numbers)

    Returns:
        list: List of n user IDs
    """
    # small sample of IDs will be common
    num_common_ids = max(int(n * 0.1), 1)

    # Generate the pool of common IDs
    common_ids = random.sample(range(min_id, max_id + 1), num_common_ids)

    # Generate the pool of uncommon IDs (the remaining possible IDs)
    all_ids = set(range(min_id, max_id + 1))
    uncommon_ids = list(all_ids - set(common_ids))

    # Generate n user IDs where 80% come from common_ids and 20% from uncommon_ids
    user_ids = []
    for _ in range(n):
        if random.random() < 0.8:  # 80% chance to pick a common ID
            user_ids.append(random.choice(common_ids))
        else:  # 20% chance to pick an uncommon ID
            user_ids.append(random.choice(uncommon_ids))

    return user_ids
