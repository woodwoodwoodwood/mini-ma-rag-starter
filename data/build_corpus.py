"""
Mini-MA-RAG: 构建示例语料库

本脚本生成一个小型维基百科风格语料库（corpus.jsonl），
供学生在没有外部数据集的情况下快速跑通实验。

语料包含约 50 篇文档，覆盖多个领域，足以支撑测试集中的多跳问题。
"""

import json
import os

CORPUS = [
    {
        "id": "doc_001",
        "title": "Kiss and Tell (1945 film)",
        "text": (
            "Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer. "
            "The film was directed by Richard Wallace and released by Columbia Pictures. "
            "Shirley Temple, who began her film career at the age of three, was one of the most popular "
            "child stars in Hollywood history. After retiring from acting, she pursued a career in diplomacy and politics."
        ),
    },
    {
        "id": "doc_002",
        "title": "Shirley Temple",
        "text": (
            "Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, "
            "and diplomat. She was the most popular child star in Hollywood during the Great Depression. "
            "After retiring from film at the age of 22, she became active in politics and diplomacy. "
            "She served as the United States Ambassador to Ghana from 1974 to 1976, "
            "and later as the United States Ambassador to Czechoslovakia from 1989 to 1992. "
            "She also served as the Chief of Protocol of the United States from 1976 to 1977."
        ),
    },
    {
        "id": "doc_003",
        "title": "Chief of Protocol of the United States",
        "text": (
            "The Chief of Protocol is an officer of the United States Department of State. "
            "The position is responsible for advising the president, vice president, and secretary of state "
            "on matters of national and international diplomatic protocol. Shirley Temple Black held this position "
            "from 1976 to 1977 under President Gerald Ford."
        ),
    },
    {
        "id": "doc_004",
        "title": "Corliss Archer",
        "text": (
            "Corliss Archer is a fictional character in a series of American short stories and films. "
            "In the 1945 film Kiss and Tell, the character was portrayed by Shirley Temple. "
            "The film was based on a play by F. Hugh Herbert and follows the comedic adventures of the teenager Corliss."
        ),
    },
    {
        "id": "doc_005",
        "title": "Chicago Fire (season 4)",
        "text": (
            "The fourth season of Chicago Fire, an American drama television series, premiered on October 13, 2015, "
            "and concluded on May 17, 2016. The season contained 23 episodes. "
            "It aired on NBC as part of the Chicago franchise created by Dick Wolf and Matt Olmstead."
        ),
    },
    {
        "id": "doc_006",
        "title": "Chicago Fire (TV series)",
        "text": (
            "Chicago Fire is an American drama television series created by Michael Brandt and Derek Haas, "
            "and produced by Dick Wolf. It premiered on NBC on October 10, 2012. "
            "The show follows the lives of the firefighters and paramedics working at Firehouse 51 of the Chicago Fire Department."
        ),
    },
    {
        "id": "doc_007",
        "title": "Peter Griffith",
        "text": (
            "Peter Griffith (1924–2001) was an American actor and advertising executive. "
            "He was the father of actress Melanie Griffith. "
            "His granddaughter is Dakota Johnson, the daughter of Melanie Griffith and Don Johnson."
        ),
    },
    {
        "id": "doc_008",
        "title": "Dakota Johnson",
        "text": (
            "Dakota Mayi Johnson (born October 4, 1989) is an American actress and model. "
            "She is the daughter of actors Don Johnson and Melanie Griffith. "
            "Her maternal grandfather is Peter Griffith. She made her screen debut in the 1999 film "
            "'Crazy in Alabama' alongside her mother."
        ),
    },
    {
        "id": "doc_009",
        "title": "Crazy in Alabama",
        "text": (
            "Crazy in Alabama is a 1999 American comedy-drama film directed by Antonio Banderas. "
            "It stars Melanie Griffith, David Morse, and Lucas Black. "
            "Dakota Johnson made her screen debut in this film, playing the role of Sondra."
        ),
    },
    {
        "id": "doc_010",
        "title": "Melanie Griffith",
        "text": (
            "Melanie Richards Griffith (born August 9, 1957) is an American actress. "
            "She is the daughter of actress Tippi Hedren and actor Peter Griffith. "
            "She has appeared in films such as Working Girl, for which she received an Academy Award nomination."
        ),
    },
    {
        "id": "doc_011",
        "title": "The House of Tears",
        "text": (
            "The House of Tears (also known as La casa de las lágrimas) is a 1915 Mexican silent film. "
            "It was directed by Fernando A. Rivero. Little is known about the director's later life, "
            "but he was active in the early Mexican cinema industry."
        ),
    },
    {
        "id": "doc_012",
        "title": "College Ranga",
        "text": (
            "College Ranga is a 1976 Indian Kannada-language film directed by Puttanna Kanagal. "
            "Puttanna Kanagal (1933–1985) was one of the most influential directors in Kannada cinema. "
            "He died in 1985 at the age of 52."
        ),
    },
    {
        "id": "doc_013",
        "title": "Fernando A. Rivero",
        "text": (
            "Fernando A. Rivero was a Mexican film director active during the silent film era. "
            "He directed The House of Tears in 1915. Records about his death are scarce, "
            "but he is believed to have died sometime in the 1940s or 1950s, though exact dates are uncertain."
        ),
    },
    {
        "id": "doc_014",
        "title": "The Stoneman Murders",
        "text": (
            "The Stoneman Murders is a 2009 Indian neo-noir thriller film directed by Manish Gupta. "
            "It is based on the real-life Stoneman serial killings that took place in Bombay in the 1980s."
        ),
    },
    {
        "id": "doc_015",
        "title": "Chandralekha (2014 film)",
        "text": (
            "Chandralekha is a 2014 Indian Kannada-language horror comedy film directed by Om Prakash Rao. "
            "It stars Chiranjeevi Sarja and Shanvi Srivastava. The film is a remake of the Telugu film Prema Katha Chitram."
        ),
    },
    {
        "id": "doc_016",
        "title": "Manish Gupta",
        "text": (
            "Manish Gupta is an Indian film director and screenwriter known for directing The Stoneman Murders (2009). "
            "He was born in India and has worked primarily in the Hindi film industry."
        ),
    },
    {
        "id": "doc_017",
        "title": "Om Prakash Rao",
        "text": (
            "Om Prakash Rao is an Indian film director who works primarily in Kannada cinema. "
            "He directed Chandralekha (2014) and has been active in the Kannada film industry for several decades."
        ),
    },
    {
        "id": "doc_018",
        "title": "Bill Cosby",
        "text": (
            "William Henry Cosby Jr. (born July 12, 1937) is an American former comedian, actor, and media personality. "
            "He was born and raised in Philadelphia, Pennsylvania, United States. "
            "Cosby began his career as a stand-up comic and later starred in the television sitcom The Cosby Show."
        ),
    },
    {
        "id": "doc_019",
        "title": "House of Cosbys",
        "text": (
            "House of Cosbys is an American animated science fiction sitcom created by Justin Roiland. "
            "It was produced for the Internet and first released in 2005. "
            "The show is a parody of Bill Cosby and features multiple clones of him. "
            "The series was produced in the United States."
        ),
    },
    {
        "id": "doc_020",
        "title": "United States Department of State",
        "text": (
            "The United States Department of State is the federal executive department responsible for the foreign policy "
            "and international relations of the United States. It was established in 1789 and is headquartered in Washington, D.C."
        ),
    },
    {
        "id": "doc_021",
        "title": "Ghana",
        "text": (
            "Ghana, officially the Republic of Ghana, is a country in West Africa. "
            "It became the first African country to achieve independence from colonial rule in 1957. "
            "Its capital is Accra. Shirley Temple Black served as the U.S. Ambassador to Ghana from 1974 to 1976."
        ),
    },
    {
        "id": "doc_022",
        "title": "Czechoslovakia",
        "text": (
            "Czechoslovakia was a landlocked country in Central Europe that existed from October 1918 until January 1993. "
            "It peacefully split into the Czech Republic and Slovakia. "
            "Shirley Temple Black served as the U.S. Ambassador to Czechoslovakia from 1989 to 1992."
        ),
    },
    {
        "id": "doc_023",
        "title": "Gerald Ford",
        "text": (
            "Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was the 38th president "
            "of the United States, serving from 1974 to 1977. He appointed Shirley Temple Black as Chief of Protocol in 1976."
        ),
    },
    {
        "id": "doc_024",
        "title": "Dick Wolf",
        "text": (
            "Richard Anthony Wolf (born December 20, 1946) is an American television producer, best known as the creator "
            "and executive producer of the Law & Order and Chicago franchises. He has won numerous Emmy Awards."
        ),
    },
    {
        "id": "doc_025",
        "title": "Antonio Banderas",
        "text": (
            "José Antonio Domínguez Bandera (born August 10, 1960), known professionally as Antonio Banderas, "
            "is a Spanish actor and director. He directed the 1999 film Crazy in Alabama starring his then-wife Melanie Griffith."
        ),
    },
    {
        "id": "doc_026",
        "title": "Working Girl",
        "text": (
            "Working Girl is a 1988 American romantic comedy-drama film directed by Mike Nichols. "
            "It stars Harrison Ford, Sigourney Weaver, and Melanie Griffith. "
            "Griffith received an Academy Award nomination for Best Actress for her role as Tess McGill."
        ),
    },
    {
        "id": "doc_027",
        "title": "Tippi Hedren",
        "text": (
            "Nathalie Kay 'Tippi' Hedren (born January 19, 1930) is an American actress and animal rights activist. "
            "She is the mother of actress Melanie Griffith and the grandmother of actress Dakota Johnson. "
            "She starred in Alfred Hitchcock's The Birds and Marnie."
        ),
    },
    {
        "id": "doc_028",
        "title": "Don Johnson",
        "text": (
            "Donnie Wayne Johnson (born December 15, 1949) is an American actor, producer, and singer. "
            "He is best known for his role as James 'Sonny' Crockett in the 1980s television series Miami Vice. "
            "He was married to Melanie Griffith and is the father of Dakota Johnson."
        ),
    },
    {
        "id": "doc_029",
        "title": "Miami Vice",
        "text": (
            "Miami Vice is an American crime drama television series created by Anthony Yerkovich and produced by Michael Mann "
            "for NBC. It starred Don Johnson and Philip Michael Thomas. The series ran for five seasons from 1984 to 1989."
        ),
    },
    {
        "id": "doc_030",
        "title": "Puttanna Kanagal",
        "text": (
            "S. R. Puttanna Kanagal (1933–1985) was an Indian film director and producer. "
            "He is considered one of the pioneers of parallel cinema in Kannada. "
            "He directed many landmark films including Gejje Pooje and Ranganayaki. He died in 1985."
        ),
    },
    {
        "id": "doc_031",
        "title": "Law & Order",
        "text": (
            "Law & Order is an American police procedural and legal drama television series created by Dick Wolf. "
            "It aired on NBC from 1990 to 2010, making it one of the longest-running live-action television series in American history."
        ),
    },
    {
        "id": "doc_032",
        "title": "NBC",
        "text": (
            "The National Broadcasting Company (NBC) is an American English-language commercial broadcast television and radio network. "
            "It is the flagship property of the NBC Entertainment division of NBCUniversal. "
            "It is headquartered in New York City."
        ),
    },
    {
        "id": "doc_033",
        "title": "Academy Awards",
        "text": (
            "The Academy Awards, also known as the Oscars, are awards for artistic and technical merit in the film industry. "
            "They are given annually by the Academy of Motion Picture Arts and Sciences (AMPAS) in Los Angeles, California."
        ),
    },
    {
        "id": "doc_034",
        "title": "Philadelphia",
        "text": (
            "Philadelphia is the largest city in the Commonwealth of Pennsylvania in the United States. "
            "It is the sixth-most-populous city in the nation. It was founded in 1682 by William Penn."
        ),
    },
    {
        "id": "doc_035",
        "title": "Washington, D.C.",
        "text": (
            "Washington, D.C., formally the District of Columbia, is the capital city and federal district of the United States. "
            "It is located on the east bank of the Potomac River. It houses the three branches of the federal government."
        ),
    },
    {
        "id": "doc_036",
        "title": "The Cosby Show",
        "text": (
            "The Cosby Show is an American television sitcom starring Bill Cosby, which aired for eight seasons on NBC from 1984 to 1992. "
            "It was the number one rated show on television for five consecutive seasons."
        ),
    },
    {
        "id": "doc_037",
        "title": "Justin Roiland",
        "text": (
            "Justin Roiland (born February 21, 1980) is an American voice actor, animator, writer, and producer. "
            "He is the co-creator and former executive producer of the Adult Swim animated sitcom Rick and Morty. "
            "He also created House of Cosbys in 2005."
        ),
    },
    {
        "id": "doc_038",
        "title": "Prema Katha Chitram",
        "text": (
            "Prema Katha Chitram is a 2013 Indian Telugu-language horror comedy film directed by J. Prabhakar Reddy. "
            "It was a commercial success and was later remade in Kannada as Chandralekha (2014)."
        ),
    },
    {
        "id": "doc_039",
        "title": "J. Prabhakar Reddy",
        "text": (
            "J. Prabhakar Reddy is an Indian film director who works in Telugu cinema. "
            "He directed the 2013 film Prema Katha Chitram."
        ),
    },
    {
        "id": "doc_040",
        "title": "Chiranjeevi Sarja",
        "text": (
            "Chiranjeevi Sarja (1984–2020) was an Indian actor who appeared in Kannada films. "
            "He starred in the 2014 film Chandralekha. He was the nephew of actor Arjun Sarja."
        ),
    },
    {
        "id": "doc_041",
        "title": "Arjun Sarja",
        "text": (
            "Srinivasa Sarja (born August 15, 1962), known professionally as Arjun Sarja, is an Indian actor, producer, and director. "
            "He primarily works in Tamil cinema and also in Telugu and Kannada films."
        ),
    },
    {
        "id": "doc_042",
        "title": "Gejje Pooje",
        "text": (
            "Gejje Pooje is a 1969 Indian Kannada-language film directed by Puttanna Kanagal. "
            "It is based on a novel of the same name by M. K. Indira. The film is considered a milestone in Kannada cinema."
        ),
    },
    {
        "id": "doc_043",
        "title": "Ranganayaki (film)",
        "text": (
            "Ranganayaki is a 1981 Indian Kannada-language film directed by Puttanna Kanagal. "
            "It stars Aarathi and Ambareesh. The film explores the life of a theater actress."
        ),
    },
    {
        "id": "doc_044",
        "title": "Aarathi",
        "text": (
            "Aarathi (born 1954) is an Indian actress who predominantly worked in Kannada cinema. "
            "She won the Karnataka State Film Award for Best Actress four times."
        ),
    },
    {
        "id": "doc_045",
        "title": "Ambareesh",
        "text": (
            "Malavalli Huchche Gowda Amarnath (1952–2018), known mononymously as Ambareesh, was an Indian actor and politician. "
            "He was a prominent lead actor in Kannada cinema and also served as a Member of Parliament."
        ),
    },
    {
        "id": "doc_046",
        "title": "Mike Nichols",
        "text": (
            "Mike Nichols (born Mikhail Igor Peschkowsky; November 6, 1931 – November 19, 2014) was an American film and "
            "theater director, producer, actor, and comedian. He directed Working Girl (1988) and The Graduate (1967)."
        ),
    },
    {
        "id": "doc_047",
        "title": "The Graduate",
        "text": (
            "The Graduate is a 1967 American romantic comedy-drama film directed by Mike Nichols. "
            "It stars Anne Bancroft, Dustin Hoffman, and Katharine Ross. "
            "The film was a massive critical and commercial success."
        ),
    },
    {
        "id": "doc_048",
        "title": "Dustin Hoffman",
        "text": (
            "Dustin Lee Hoffman (born August 8, 1937) is an American actor. He is one of the most acclaimed actors in Hollywood history. "
            "He has won two Academy Awards and has been nominated for five more."
        ),
    },
    {
        "id": "doc_049",
        "title": "Anne Bancroft",
        "text": (
            "Anna Maria Louisa Italiano (September 17, 1931 – June 6, 2005), known professionally as Anne Bancroft, "
            "was an American actress. She won an Academy Award for The Miracle Worker (1962)."
        ),
    },
    {
        "id": "doc_050",
        "title": "The Miracle Worker (1962 film)",
        "text": (
            "The Miracle Worker is a 1962 American biographical film about Anne Sullivan, tutor to Helen Keller. "
            "It stars Anne Bancroft as Sullivan and Patty Duke as Keller. "
            "Bancroft won the Academy Award for Best Actress for this role."
        ),
    },
]


def build():
    """生成语料库文件"""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "corpus.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in CORPUS:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[INFO] 语料库构建完成: {output_path}")
    print(f"[INFO] 共 {len(CORPUS)} 篇文档")


if __name__ == "__main__":
    build()
