import json

def load_adaptive_learning_dataset(filepath=""):
    """
    Loads the adaptive learning dataset from JSON file and formats it
    into the structure required by AdaptiveLearningEnv.
    """

    with open(filepath, "r") as f:
        data = json.load(f)

    lessons = {}
    for lesson in data["lessons"]:
        lessons[lesson["lesson_id"]] = {
            "required_mastery_level": lesson["required_mastery_level"],
            "prerequisites": lesson["prerequisites"]
        }

    activities = {}
    for activity in data["activities"]:
        activities[activity["activity_id"]] = {
            "lesson_contributions": activity["lesson_contributions"],
            "style": activity["style"],
            "difficulty": activity["difficulty"],
            "max_selection_limit": activity["max_selection_limit"]
        }

    sprints = data["sprints"]
    # Convert sprint keys to integers
    sprints = {int(k): v for k, v in sprints.items()}

    return lessons, activities, sprints



