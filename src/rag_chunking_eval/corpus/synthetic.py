"""Deterministic synthetic corpus for chunking evaluation.

Design
------
Twenty short documents on everyday topics (space, cooking, history, sports,
technology, music, medicine, architecture, oceans, birds, railways, cinema,
literature, gardening, geology, economics, aviation, archaeology, linguistics,
robotics). Each document is 8-10 sentences long.

Each document plants exactly two distinctive facts built around unique proper
nouns (a name, place, or thing that appears nowhere else in the corpus), and
asks two questions, one per fact. The gold answer is a short span taken from
the fact sentence, and gold_sentence_ids point at that sentence.

Every document also carries distractor sentences: they reuse the question's
vocabulary (years, names of similar things, the same units) but do NOT
contain the answer. A retrieval pass that keys on loose keywords will pull
the distractor; only a chunker that keeps the fact sentence intact lets the
pipeline score full recall. Naive chunking (splitting mid-fact or dropping
the fact sentence) gets punished.

Everything is seeded: a random.Random(seed) shuffles filler sentence order,
so the same seed always builds the same corpus.

Question dict keys: question_id ("syn-q000".."syn-q039"), doc_id, question,
answer, gold_sentence_ids (["{doc_id}:s{j}", ...], j = sentence index in the
document as produced by text.split_sentences).
"""

from __future__ import annotations

import random

# Each topic: six filler sentences, two facts (sentence + question + answer),
# and two distractors. The doc builder interleaves them into 10 sentences.
_TOPICS = [
    dict(
        topic="astronomy",
        filler=[
            "Astronomy is one of the oldest sciences humans have practiced.",
            "Early stargazers tracked the planets with nothing but their eyes.",
            "Modern telescopes collect far more light than the human eye.",
            "Clear mountain air gives observatories their best viewing nights.",
            "Students often visit the planetarium on school field trips.",
            "Star charts help beginners learn the constellations quickly.",
        ],
        facts=[
            dict(
                sentence="The Meridian telescope was installed at Alderwick Observatory in 1847.",
                question="In what year was the Meridian telescope installed?",
                answer="1847",
            ),
            dict(
                sentence="Vela Corrigan discovered the comet Corrigan-9 in the spring of 1963.",
                question="Who discovered the comet Corrigan-9?",
                answer="Vela Corrigan",
            ),
        ],
        distractors=[
            "Astronomers still debate whether the Kessler telescope, installed in 1911, copied the Meridian design.",
            "The comet was named for the observatory director who funded the survey, not for its discoverer.",
        ],
    ),
    dict(
        topic="cooking",
        filler=[
            "Good cooking starts with fresh ingredients and sharp knives.",
            "Many chefs taste their sauces several times while they cook.",
            "Salt added early builds deeper flavor in stews and soups.",
            "Home cooks often underestimate how hot their ovens run.",
            "A well-seasoned pan makes delicate fish much easier to flip.",
            "Resting meat after cooking keeps the juices inside the cut.",
        ],
        facts=[
            dict(
                sentence="Chef Amara Diallo created the dish poulet yassa blanc in 2009.",
                question="In what year did Amara Diallo create poulet yassa blanc?",
                answer="2009",
            ),
            dict(
                sentence="The saffron broth at Maison Lierre simmers for exactly eleven hours.",
                question="How long does the saffron broth at Maison Lierre simmer?",
                answer="eleven hours",
            ),
        ],
        distractors=[
            "Her 2015 tasting menu won a regional prize, but it did not include the famous chicken dish.",
            "The regular stock at Maison Lierre simmers for only three hours before service.",
        ],
    ),
    dict(
        topic="history",
        filler=[
            "History is usually written by the people who kept the records.",
            "Old letters can reveal details that official chronicles omit.",
            "Museums preserve everyday objects alongside royal treasures.",
            "Timelines help students see how events connect across centuries.",
            "Primary sources are the backbone of serious historical research.",
            "Many medieval towns still follow their original street plans.",
        ],
        facts=[
            dict(
                sentence="The Treaty of Halloway was signed on the 14th of March 1812.",
                question="On what date was the Treaty of Halloway signed?",
                answer="14th of March 1812",
            ),
            dict(
                sentence="Lighthouse keeper Bram Okafor kept the Eddystone light burning for 41 consecutive nights during the great storm.",
                question="How many consecutive nights did Bram Okafor keep the light burning?",
                answer="41",
            ),
        ],
        distractors=[
            "A later armistice in 1814 is often confused with the Treaty of Halloway.",
            "His assistant managed only nine nights alone before the relief crew arrived.",
        ],
    ),
    dict(
        topic="sports",
        filler=[
            "Endurance athletes train for months before a major race.",
            "Team sports reward communication as much as raw talent.",
            "Warm-up routines reduce the risk of muscle injuries.",
            "Crowds make a real difference in close championship games.",
            "Young players benefit most from consistent practice schedules.",
            "Recovery days are just as important as hard training days.",
        ],
        facts=[
            dict(
                sentence="Kenji Mori set the Brackenwood marathon record of 2 hours 6 minutes in 2019.",
                question="In what year did Kenji Mori set the Brackenwood marathon record?",
                answer="2019",
            ),
            dict(
                sentence="The Harrowgate Cup final drew a crowd of 68,400 spectators.",
                question="How many spectators attended the Harrowgate Cup final?",
                answer="68,400",
            ),
        ],
        distractors=[
            "The previous record from 2011 stood for eight years before anyone broke it.",
            "The semifinal drew only 31,000 spectators the week before.",
        ],
    ),
    dict(
        topic="technology",
        filler=[
            "New gadgets usually improve a little each year rather than a lot.",
            "Battery life matters more to buyers than raw processor speed.",
            "Software updates can extend the life of older hardware.",
            "Tech reviewers test devices under real daily conditions.",
            "Most people charge their phones overnight without thinking about it.",
            "Repair shops see the same cracked screens every single week.",
        ],
        facts=[
            dict(
                sentence="The Quillpad tablet was the first consumer device to ship with a haptic stylus in 2021.",
                question="In what year did the Quillpad tablet ship with a haptic stylus?",
                answer="2021",
            ),
            dict(
                sentence="Nerovolt batteries retain 94 percent of their charge after 1,000 cycles.",
                question="What percentage of charge do Nerovolt batteries retain after 1,000 cycles?",
                answer="94 percent",
            ),
        ],
        distractors=[
            "A rival tablet added a haptic stylus in 2023, two years after the first.",
            "Standard lithium cells in the same test retained only 80 percent after 1,000 cycles.",
        ],
    ),
    dict(
        topic="music",
        filler=[
            "Learning an instrument takes patience more than talent.",
            "Live concerts carry an energy that recordings cannot match.",
            "Many famous songs were written in a single afternoon.",
            "Choirs blend dozens of voices into one unified sound.",
            "Sheet music lets musicians share compositions across centuries.",
            "Street performers often draw the most honest crowds in a city.",
        ],
        facts=[
            dict(
                sentence="The Velvet Comets recorded their debut album in a converted grain silo in 1998.",
                question="Where did The Velvet Comets record their debut album?",
                answer="a converted grain silo",
            ),
            dict(
                sentence="Luthier Ines Halvorsen carves each violin bridge from a single piece of Balkan maple.",
                question="What wood does Ines Halvorsen use for violin bridges?",
                answer="Balkan maple",
            ),
        ],
        distractors=[
            "Their second album was recorded in a proper studio in Nashville.",
            "Her student uses cheaper spruce for practice instruments.",
        ],
    ),
    dict(
        topic="medicine",
        filler=[
            "Vaccines have saved more lives than any other medical advance.",
            "Hand washing remains the simplest infection control measure.",
            "Clinical trials test new treatments in careful stages.",
            "Nurses often notice patient changes before the monitors do.",
            "Regular checkups catch many problems while they are small.",
            "Medical records help doctors see the full patient history.",
        ],
        facts=[
            dict(
                sentence="Lena Osei published the first trial of the Renexa vaccine in 2016.",
                question="In what year did Lena Osei publish the first Renexa vaccine trial?",
                answer="2016",
            ),
            dict(
                sentence="The portable dialysis unit weighs just 3.2 kilograms.",
                question="How much does the portable dialysis unit weigh?",
                answer="3.2 kilograms",
            ),
        ],
        distractors=[
            "A competing team published a smaller vaccine trial in 2018.",
            "The hospital model it replaced weighed over 40 kilograms.",
        ],
    ),
    dict(
        topic="architecture",
        filler=[
            "Great buildings balance beauty with practical use.",
            "Old stone walls keep interiors cool in hot summers.",
            "Architects sketch dozens of drafts before settling on a plan.",
            "Public squares shape how a city feels to walk through.",
            "Restoring historic facades takes skilled craftspeople.",
            "Natural light changes how a room feels through the day.",
        ],
        facts=[
            dict(
                sentence="The Halden Tower in Oslo was completed in 1974.",
                question="When was the Halden Tower completed?",
                answer="1974",
            ),
            dict(
                sentence="Architect Tomas Revik designed the spiral library with no interior columns.",
                question="Who designed the spiral library with no interior columns?",
                answer="Tomas Revik",
            ),
        ],
        distractors=[
            "The neighboring Fjeld House was completed much later, in 1989.",
            "The city rejected his first proposal, which had twelve columns.",
        ],
    ),
    dict(
        topic="oceans",
        filler=[
            "Oceans cover more than two thirds of the planet's surface.",
            "Tides rise and fall with the pull of the moon.",
            "Coral reefs support a quarter of all marine species.",
            "Deep sea creatures live in complete darkness.",
            "Sailors have navigated by the stars for thousands of years.",
            "Coastal towns depend on fishing for their livelihood.",
        ],
        facts=[
            dict(
                sentence="The research vessel Pelagia mapped the Sorn Trench to a depth of 7,200 meters.",
                question="To what depth did the Pelagia map the Sorn Trench?",
                answer="7,200 meters",
            ),
            dict(
                sentence="Biologist Mara Voss identified the ghost octopus at a depth of 4,100 meters.",
                question="At what depth did Mara Voss identify the ghost octopus?",
                answer="4,100 meters",
            ),
        ],
        distractors=[
            "An earlier survey in 2009 only reached 5,000 meters.",
            "A different team found a related species much shallower, at 900 meters.",
        ],
    ),
    dict(
        topic="birds",
        filler=[
            "Birds navigate by the sun, stars, and the earth's magnetic field.",
            "Many songbirds learn their calls by listening to their parents.",
            "Migration lets birds follow food across the seasons.",
            "Feathers provide both insulation and the shape needed for flight.",
            "Urban pigeons have adapted to city life remarkably well.",
            "Birdwatchers keep careful lists of every species they spot.",
        ],
        facts=[
            dict(
                sentence="The wandering albatross can glide for six hours without a single wingbeat.",
                question="How long can the wandering albatross glide without a wingbeat?",
                answer="six hours",
            ),
            dict(
                sentence="Ornithologist Priya Nair banded 1,240 Arctic terns in a single season.",
                question="How many Arctic terns did Priya Nair band in a single season?",
                answer="1,240",
            ),
        ],
        distractors=[
            "The smaller gull must flap every few minutes to stay aloft.",
            "Her colleague banded only 300 terns that same season.",
        ],
    ),
    dict(
        topic="railways",
        filler=[
            "Trains move more freight per gallon of fuel than trucks.",
            "Railway stations were once the busiest places in every town.",
            "Engineers inspect the tracks regularly for cracks and wear.",
            "Night trains let travelers sleep through long journeys.",
            "Timetables coordinate thousands of departures every day.",
            "Steam locomotives gave way to diesel and then electric power.",
        ],
        facts=[
            dict(
                sentence="The Nordspan line opened in 1888, linking Bergen to the inland mines.",
                question="In what year did the Nordspan line open?",
                answer="1888",
            ),
            dict(
                sentence="Engineer Gustav Lindt designed the viaduct with nineteen stone arches.",
                question="How many stone arches does Gustav Lindt's viaduct have?",
                answer="nineteen",
            ),
        ],
        distractors=[
            "The coastal branch of the network opened decades later, in 1924.",
            "The earlier wooden trestle had only seven spans.",
        ],
    ),
    dict(
        topic="cinema",
        filler=[
            "Early films had no sound and relied on live piano players.",
            "A good script is the foundation of every great movie.",
            "Lighting crews shape the mood of each scene carefully.",
            "Film festivals launch the careers of new directors.",
            "Audiences remember strong endings more than strong openings.",
            "Classic movies are still screened in revival theaters.",
        ],
        facts=[
            dict(
                sentence="Director Sofia Marchetti shot Paper Lanterns in twenty-three days.",
                question="How many days did Sofia Marchetti take to shoot Paper Lanterns?",
                answer="twenty-three days",
            ),
            dict(
                sentence="The film won the Golden Ibis at the 2004 Marbella festival.",
                question="Which award did the film win at the 2004 Marbella festival?",
                answer="the Golden Ibis",
            ),
        ],
        distractors=[
            "The studio expected the shoot to last at least forty days.",
            "It lost the audience prize to a documentary that year.",
        ],
    ),
    dict(
        topic="literature",
        filler=[
            "Novels let readers live many lives in a single sitting.",
            "Libraries lend millions of books to their communities.",
            "Poets compress big feelings into very few words.",
            "Book clubs turn solitary reading into shared conversation.",
            "Translators carry stories across language barriers.",
            "First editions can become valuable collector's items.",
        ],
        facts=[
            dict(
                sentence="Novelist Adaeze Kalu wrote The Salt Road over nine years.",
                question="How long did Adaeze Kalu take to write The Salt Road?",
                answer="nine years",
            ),
            dict(
                sentence="The Thornfield Prize carries an award of 50,000 pounds.",
                question="How much is the Thornfield Prize award?",
                answer="50,000 pounds",
            ),
        ],
        distractors=[
            "Her first novel took only eighteen months to complete.",
            "The shortlist prize is a token 1,000 pounds.",
        ],
    ),
    dict(
        topic="gardening",
        filler=[
            "Healthy soil is the real secret behind thriving plants.",
            "Most vegetables need full sun for at least half the day.",
            "Compost turns kitchen scraps into rich garden food.",
            "Watering in the morning reduces fungal disease.",
            "Bees do most of the pollination work in a home garden.",
            "Pruning in late winter encourages strong spring growth.",
        ],
        facts=[
            dict(
                sentence="Botanist Ruth Adler bred the Midnight Ember rose over fourteen seasons.",
                question="How many seasons did Ruth Adler take to breed the Midnight Ember rose?",
                answer="fourteen",
            ),
            dict(
                sentence="The rose needs exactly six hours of morning sun each day.",
                question="How much morning sun does the Midnight Ember rose need?",
                answer="six hours",
            ),
        ],
        distractors=[
            "The commercial version was rushed to market after only three seasons of testing.",
            "Afternoon shade is optional but helps the blooms last longer.",
        ],
    ),
    dict(
        topic="geology",
        filler=[
            "Rocks record the history of the earth in their layers.",
            "Erosion carves valleys slowly over millions of years.",
            "Geologists read cliffs the way others read books.",
            "Earthquakes release energy stored in stressed rock.",
            "Fossils mark the chapters of ancient life.",
            "Glaciers grind the landscape as they creep downhill.",
        ],
        facts=[
            dict(
                sentence="Mount Verra last erupted in 1739, burying two villages in ash.",
                question="When did Mount Verra last erupt?",
                answer="1739",
            ),
            dict(
                sentence="The mineral kallite glows faint green under ultraviolet light.",
                question="What color does kallite glow under ultraviolet light?",
                answer="green",
            ),
        ],
        distractors=[
            "A smaller eruption in 1902 caused no damage.",
            "The similar-looking mineral sorlite glows orange instead.",
        ],
    ),
    dict(
        topic="economics",
        filler=[
            "Prices rise when demand outruns supply.",
            "Central banks adjust interest rates to steer the economy.",
            "Inflation quietly shrinks what a paycheck can buy.",
            "Trade lets countries specialize in what they do best.",
            "Economists study both markets and human behavior.",
            "Recessions test the resilience of households and firms.",
        ],
        facts=[
            dict(
                sentence="Economist David Okonkwo proposed the Harbor Index in 1994.",
                question="In what year did David Okonkwo propose the Harbor Index?",
                answer="1994",
            ),
            dict(
                sentence="The index tracks the prices of exactly forty commodities.",
                question="How many commodities does the Harbor Index track?",
                answer="forty",
            ),
        ],
        distractors=[
            "A rival index launched in 2001 never gained the same following.",
            "An older index tracked only twelve commodities.",
        ],
    ),
    dict(
        topic="aviation",
        filler=[
            "Pilots train for hundreds of hours before flying solo.",
            "Air traffic controllers keep crowded skies safe.",
            "Jet engines are most efficient at high cruising altitudes.",
            "Weather briefings are a mandatory part of flight planning.",
            "Small planes can land on grass strips larger jets cannot use.",
            "Flight simulators let crews practice emergencies safely.",
        ],
        facts=[
            dict(
                sentence="Pilot Ingrid Solberg flew the first solar crossing of the Atlantic in 2013.",
                question="In what year did Ingrid Solberg fly the first solar Atlantic crossing?",
                answer="2013",
            ),
            dict(
                sentence="The aircraft cruised at 8,500 meters for most of the flight.",
                question="At what altitude did the aircraft cruise?",
                answer="8,500 meters",
            ),
        ],
        distractors=[
            "A gas-balloon crossing in 1999 took nearly twice as long.",
            "It climbed to 10,000 meters only to avoid a storm.",
        ],
    ),
    dict(
        topic="archaeology",
        filler=[
            "Archaeologists dig in thin layers to preserve context.",
            "Pottery shards are the most common finds at ancient sites.",
            "Carbon dating reveals the age of organic remains.",
            "Field notes matter as much as the artifacts themselves.",
            "Many sites are found by accident during construction.",
            "Museums display only a fraction of what excavations recover.",
        ],
        facts=[
            dict(
                sentence="The dig at Tel Maresh uncovered a 3,000-year-old olive press.",
                question="How old was the olive press found at Tel Maresh?",
                answer="3,000 years old",
            ),
            dict(
                sentence="Archaeologist Yusuf Haddad catalogued 212 pottery shards in one week.",
                question="How many pottery shards did Yusuf Haddad catalogue in one week?",
                answer="212",
            ),
        ],
        distractors=[
            "A Roman-era press nearby dated to only 1,800 years ago.",
            "The previous team managed just 60 shards in the same time.",
        ],
    ),
    dict(
        topic="linguistics",
        filler=[
            "Children absorb grammar rules without formal lessons.",
            "Languages borrow words from each other constantly.",
            "Accents shift a little with every generation.",
            "Writing systems took centuries to develop.",
            "Bilingual speakers switch languages without noticing.",
            "Field linguists record endangered languages before they vanish.",
        ],
        facts=[
            dict(
                sentence="Scholar Elena Petrova documented the last native speaker of Livvian in 1987.",
                question="In what year did Elena Petrova document the last native Livvian speaker?",
                answer="1987",
            ),
            dict(
                sentence="Livvian has only eleven distinct vowel sounds.",
                question="How many distinct vowel sounds does Livvian have?",
                answer="eleven",
            ),
        ],
        distractors=[
            "A related dialect was documented as late as 2005.",
            "Its consonant inventory is much larger, at thirty sounds.",
        ],
    ),
    dict(
        topic="robotics",
        filler=[
            "Robots excel at tasks that are dull, dirty, or dangerous.",
            "Sensors give machines a rough sense of their surroundings.",
            "Factory robots repeat the same motion thousands of times.",
            "Engineers test new robots in simulation before real trials.",
            "Battery weight is the main limit on mobile robots.",
            "Simple grippers handle most warehouse picking jobs.",
        ],
        facts=[
            dict(
                sentence="The Halcyon Lab's robot Kestrel-7 assembled a bicycle in 4 minutes 12 seconds.",
                question="How fast did Kestrel-7 assemble a bicycle?",
                answer="4 minutes 12 seconds",
            ),
            dict(
                sentence="Kestrel-7 uses a torque sensor accurate to 0.02 newton-meters.",
                question="How accurate is Kestrel-7's torque sensor?",
                answer="0.02 newton-meters",
            ),
        ],
        distractors=[
            "The previous model needed over nine minutes for the same task.",
            "Its gripper camera resolves details down to half a millimeter.",
        ],
    ),
]


def build_synthetic_corpus(seed: int = 7) -> tuple[dict[str, str], list[dict]]:
    """Build the synthetic corpus and its 40 questions.

    Returns (docs, questions) where docs maps "syn-doc-00".."syn-doc-19" to
    document text and each question dict has question_id, doc_id, question,
    answer, and gold_sentence_ids. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    docs: dict[str, str] = {}
    questions: list[dict] = []
    q_index = 0
    for i, topic in enumerate(_TOPICS):
        doc_id = f"syn-doc-{i:02d}"
        filler = list(topic["filler"])
        rng.shuffle(filler)
        # Interleave: filler, fact1, distractor1, filler, fact2, distractor2, filler.
        seq: list[str] = []
        gold: list[str] = []
        seq.extend(filler[:2])
        for fact, distractor in zip(topic["facts"], topic["distractors"]):
            gold.append(len(seq))
            seq.append(fact["sentence"])
            seq.append(distractor)
            seq.extend(filler[2:4] if len(gold) == 1 else filler[4:])
        docs[doc_id] = " ".join(seq)
        for fact, s_idx in zip(topic["facts"], gold):
            questions.append(
                {
                    "question_id": f"syn-q{q_index:03d}",
                    "doc_id": doc_id,
                    "question": fact["question"],
                    "answer": fact["answer"],
                    "gold_sentence_ids": [f"{doc_id}:s{s_idx}"],
                }
            )
            q_index += 1
    return docs, questions
