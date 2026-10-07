"""Creates data/sample_hotpot.json: a tiny hand-written file in EXACTLY the HotpotQA
format, used only to test the pipeline when the real dataset is not downloaded.
Real results in the report must come from hotpot_dev_distractor_v1.json."""
import json
import random

P = {
 "Interstellar (film)": "Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. It stars Matthew McConaughey, Anne Hathaway and Jessica Chastain. Set in a dystopian future, the film follows a group of astronauts who travel through a wormhole.",
 "Christopher Nolan": "Christopher Edward Nolan is a British and American filmmaker. Born in London, Nolan studied English literature at University College London. He is known for Memento, The Dark Knight trilogy and Inception.",
 "Inception": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, starring Leonardo DiCaprio as a professional thief who steals information by infiltrating dreams.",
 "Denis Villeneuve": "Denis Villeneuve is a Canadian filmmaker. He is a four-time recipient of the Canadian Screen Award for Best Direction. He directed Arrival, Blade Runner 2049 and Dune.",
 "Arrival (film)": "Arrival is a 2016 American science fiction drama film directed by Denis Villeneuve. It stars Amy Adams as a linguist who is recruited to communicate with aliens.",
 "Matthew McConaughey": "Matthew David McConaughey is an American actor. He was born in Uvalde, Texas, and graduated from the University of Texas at Austin with a degree in radio-television-film.",
 "University College London": "University College London is a public research university in London, England. It was founded in 1826 and is a member institution of the federal University of London.",
 "Scott Derrickson": "Scott Derrickson is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films and the Marvel film Doctor Strange.",
 "Ed Wood": "Edward Davis Wood Jr. was an American filmmaker, actor, and author. In the 1950s, Wood made several low-budget science fiction, crime and horror films.",
 "Arthur's Magazine": "Arthur's Magazine was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it was started in 1844.",
 "First for Women": "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.",
 "Amy Adams": "Amy Lou Adams is an American actress. She was born in Vicenza, Italy, to American parents and was raised in Castle Rock, Colorado.",
 "Leonardo DiCaprio": "Leonardo Wilhelm DiCaprio is an American actor and film producer born in Los Angeles. He won the Academy Award for Best Actor for The Revenant.",
 "Blade Runner 2049": "Blade Runner 2049 is a 2017 American science fiction film directed by Denis Villeneuve, starring Ryan Gosling and Harrison Ford.",
 "Satyajit Ray": "Satyajit Ray was an Indian film director, writer and illustrator. He was born in Calcutta into a Bengali family and directed thirty-six films.",
 "Pather Panchali": "Pather Panchali is a 1955 Indian Bengali-language drama film written and directed by Satyajit Ray. It was the first film of The Apu Trilogy.",
 "Rabindranath Tagore": "Rabindranath Tagore was a Bengali poet, writer and composer from Calcutta. In 1921 he founded Visva-Bharati University in Santiniketan.",
 "Visva-Bharati University": "Visva-Bharati is a public central university located in Santiniketan, West Bengal, India. It was founded by Rabindranath Tagore.",
 "Gitanjali": "Gitanjali is a collection of poems by the Bengali poet Rabindranath Tagore. The English translation won the Nobel Prize in Literature in 1913.",
 "Python (programming language)": "Python is a high-level, general-purpose programming language. It was created by Guido van Rossum and first released in 1991.",
 "Guido van Rossum": "Guido van Rossum is a Dutch programmer. He received a master's degree in mathematics and computer science from the University of Amsterdam in 1982.",
 "Java (programming language)": "Java is a high-level, class-based, object-oriented programming language. It was originally developed by James Gosling at Sun Microsystems and released in 1995.",
 "James Gosling": "James Gosling is a Canadian computer scientist, best known as the founder and lead designer behind the Java programming language. He earned a PhD from Carnegie Mellon University.",
 "Sun Microsystems": "Sun Microsystems was an American technology company headquartered in Santa Clara, California. It was acquired by Oracle Corporation in 2010.",
 "Dune (2021 film)": "Dune is a 2021 American epic science fiction film directed by Denis Villeneuve, starring Timothée Chalamet.",
 "Kolkata": "Kolkata, formerly Calcutta, is the capital of the Indian state of West Bengal. It is home to the Indian Institute of Management Calcutta.",
}

Q = [
 ("Which university did the director of Interstellar study at?", "University College London", "bridge", ["Interstellar (film)", "Christopher Nolan"]),
 ("Were Scott Derrickson and Ed Wood of the same nationality?", "yes", "comparison", ["Scott Derrickson", "Ed Wood"]),
 ("Which magazine was started first, Arthur's Magazine or First for Women?", "Arthur's Magazine", "comparison", ["Arthur's Magazine", "First for Women"]),
 ("In which country was the lead actress of Arrival born?", "Italy", "bridge", ["Arrival (film)", "Amy Adams"]),
 ("From which university did the creator of the Python programming language get his degree?", "University of Amsterdam", "bridge", ["Python (programming language)", "Guido van Rossum"]),
 ("Where was the company that originally developed Java headquartered?", "Santa Clara, California", "bridge", ["Java (programming language)", "Sun Microsystems"]),
 ("Which university was founded by the author of Gitanjali?", "Visva-Bharati University", "bridge", ["Gitanjali", "Rabindranath Tagore"]),
 ("Are Denis Villeneuve and Christopher Nolan from the same country?", "no", "comparison", ["Denis Villeneuve", "Christopher Nolan"]),
 ("In which city was the director of Pather Panchali born?", "Calcutta", "bridge", ["Pather Panchali", "Satyajit Ray"]),
 ("Which university did the lead actor of Interstellar graduate from?", "University of Texas at Austin", "bridge", ["Interstellar (film)", "Matthew McConaughey"]),
]


def main():
    rng = random.Random(0)
    titles = list(P)
    out = []
    for i, (q, a, typ, gold) in enumerate(Q):
        others = rng.sample([t for t in titles if t not in gold], 8)
        ctx = [[t, [s + "." for s in P[t].rstrip(".").split(". ")]] for t in gold + others]
        rng.shuffle(ctx)
        out.append({"_id": f"sample{i}", "question": q, "answer": a, "type": typ,
                    "level": "medium", "supporting_facts": [[g, 0] for g in gold],
                    "context": ctx})
    with open("data/sample_hotpot.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote data/sample_hotpot.json with", len(out), "questions")


if __name__ == "__main__":
    main()
