few_shots = [
    {
        "Question": "How many users are there in total?",
        "SQLQuery": "SELECT COUNT(`id`) AS total_users FROM `users`;",
        "SQLResult": [(100,)],
        "Answer": "There are 100 users in total."
    },
    {
        "Question": "How many photos have been uploaded in total?",
        "SQLQuery": "SELECT COUNT(`id`) AS total_photos FROM `photos`;",
        "SQLResult": [(500,)],
        "Answer": "There are 500 photos uploaded in total."
    },
    {
        "Question": "Which user has uploaded the most photos?",
        "SQLQuery": (
            "SELECT username, COUNT(*) AS total_uploads "
            "FROM users "
            "JOIN photos ON users.id = photos.user_id "
            "GROUP BY username "
            "ORDER BY total_uploads DESC "
            "LIMIT 1;"
        ),
        "SQLResult": [("john_doe", 45)],
        "Answer": "The user with the most photos is john_doe with 45 uploads."
    },
    {
        "Question": "How many likes does photo with id 10 have?",
        "SQLQuery": "SELECT COUNT(`id`) AS like_count FROM `likes` WHERE `photo_id` = 10;",
        "SQLResult": [(25,)],
        "Answer": "The photo with ID 10 has 25 likes."
    },
    {
        "Question": "How many comments are there in total?",
        "SQLQuery": "SELECT COUNT(`id`) AS total_comments FROM `comments`;",
        "SQLResult": [(300,)],
        "Answer": "There are 300 comments in total."
    },
    {
        "Question": "Which photo has the most likes?",
        "SQLQuery": (
            "SELECT p.`id`, COUNT(l.`id`) AS like_count "
            "FROM photos p "
            "JOIN likes l ON p.`id` = l.`photo_id` "
            "GROUP BY p.`id` "
            "ORDER BY like_count DESC "
            "LIMIT 1;"
        ),
        "SQLResult": [(15, 120)],
        "Answer": "The photo with ID 15 has the most likes (120)."
    },
    {
        "Question": "How many followers does user with id 5 have?",
        "SQLQuery": "SELECT COUNT(`follower_id`) AS follower_count FROM `follows` WHERE `followee_id` = 5;",
        "SQLResult": [(40,)],
        "Answer": "The user with ID 5 has 40 followers."
    },
    {
        "Question": "What are the names of the first 5 tags?",
        "SQLQuery": "SELECT `tag_name` FROM `tags` ORDER BY `created_at` ASC LIMIT 5;",
        "SQLResult": [('travel',), ('food',), ('nature',), ('fashion',), ('sports',)],
        "Answer": "The first 5 tags are travel, food, nature, fashion, and sports."
    }
]
