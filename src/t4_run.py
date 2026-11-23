"""T4 pilot, step 1: collect LLM scores on GSS items (2016-2022 pooled).
Forward: p_LLM(answer | group persona) and p_LLM(answer | no persona).
Inverse: p_LLM(group | question, answer) and p_LLM(group | question) for 7 demographic partitions.
~1,700 score_options calls. Output: t4_llm.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, time
sys.path.insert(0, (_REPO + '/outputs/screen'))
from llm import score_options

NAT = "We are faced with many problems in this country, none of which can be solved easily or inexpensively. Are we spending too much, too little, or about the right amount on {}?"
NATO = ['Too little', 'About right', 'Too much']
SEXO = ['Always wrong', 'Almost always wrong', 'Wrong only sometimes', 'Not wrong at all']
CON = "I am going to name some institutions in this country. As far as the people running this institution are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them? Institution: {}."
CONO = ['A great deal', 'Only some', 'Hardly any']
# item: (question, option labels, GSS codes in the same order)
ITEMS = {
 'grass': ("Do you think the use of marijuana should be made legal or not?", ['Should be legal', 'Should not be legal'], [1, 2]),
 'abany': ("Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if the woman wants it for any reason.", ['Yes', 'No'], [1, 2]),
 'cappun': ("Do you favor or oppose the death penalty for persons convicted of murder?", ['Favor', 'Oppose'], [1, 2]),
 'gunlaw': ("Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?", ['Favor', 'Oppose'], [1, 2]),
 'homosex': ("What about sexual relations between two adults of the same sex -- do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?", SEXO, [1, 2, 3, 4]),
 'premarsx': ("If a man and woman have sex relations before marriage, do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?", SEXO, [1, 2, 3, 4]),
 'xmarsex': ("What is your opinion about a married person having sexual relations with someone other than the marriage partner -- is it always wrong, almost always wrong, wrong only sometimes, or not wrong at all?", SEXO, [1, 2, 3, 4]),
 'teensex': ("What if they are in their early teens, say 14 to 16 years old? In that case, do you think sex relations before marriage are always wrong, almost always wrong, wrong only sometimes, or not wrong at all?", SEXO, [1, 2, 3, 4]),
 'fepol': ("Tell me if you agree or disagree with this statement: Most men are better suited emotionally for politics than are most women.", ['Agree', 'Disagree'], [1, 2]),
 'fefam': ("Do you agree or disagree: It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family.", ['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree'], [1, 2, 3, 4]),
 'spkath': ("There are always some people whose ideas are considered bad or dangerous by other people. For instance, somebody who is against all churches and religion. If such a person wanted to make a speech in your community against churches and religion, should he be allowed to speak, or not?", ['Yes, allowed to speak', 'Not allowed'], [1, 2]),
 'colath': ("Consider somebody who is against all churches and religion. Should such a person be allowed to teach in a college or university, or not?", ['Yes, allowed to teach', 'Not allowed'], [4, 5]),
 'libath': ("Consider somebody who is against all churches and religion. If some people in your community suggested that a book he wrote against churches and religion should be taken out of your public library, would you favor removing this book, or not?", ['Favor removing the book', 'Not favor removing the book'], [1, 2]),
 'natenvir': (NAT.format('improving and protecting the environment'), NATO, [1, 2, 3]),
 'natheal': (NAT.format("improving and protecting the nation's health"), NATO, [1, 2, 3]),
 'natrace': (NAT.format('improving the conditions of Blacks'), NATO, [1, 2, 3]),
 'natfare': (NAT.format('welfare'), NATO, [1, 2, 3]),
 'natarms': (NAT.format('the military, armaments and defense'), NATO, [1, 2, 3]),
 'natcrime': (NAT.format('halting the rising crime rate'), NATO, [1, 2, 3]),
 'trust': ("Generally speaking, would you say that most people can be trusted or that you can't be too careful in dealing with people?", ['Most people can be trusted', "Can't be too careful"], [1, 2]),
 'fair': ("Do you think most people would try to take advantage of you if they got a chance, or would they try to be fair?", ['Would take advantage of you', 'Would try to be fair'], [1, 2]),
 'helpful': ("Would you say that most of the time people try to be helpful, or that they are mostly just looking out for themselves?", ['Try to be helpful', 'Just look out for themselves'], [1, 2]),
 'conarmy': (CON.format('the military'), CONO, [1, 2, 3]),
 'conpress': (CON.format('the press'), CONO, [1, 2, 3]),
 'confed': (CON.format('the executive branch of the federal government'), CONO, [1, 2, 3]),
 'postlife': ("Do you believe there is a life after death?", ['Yes', 'No'], [1, 2]),
 'prayer': ("The United States Supreme Court has ruled that no state or local government may require the reading of the Lord's Prayer or Bible verses in public schools. What are your views on this -- do you approve or disapprove of the court ruling?", ['Approve', 'Disapprove'], [1, 2]),
 'bible': ("Which of these statements comes closest to describing your feelings about the Bible?", ['The Bible is the actual word of God and is to be taken literally, word for word', 'The Bible is the inspired word of God but not everything in it should be taken literally', 'The Bible is an ancient book of fables, legends, history, and moral precepts recorded by men'], [1, 2, 3]),
 'letdie1': ("When a person has a disease that cannot be cured, do you think doctors should be allowed by law to end the patient's life by some painless means if the patient and his family request it?", ['Yes', 'No'], [1, 2]),
 'suicide1': ("Do you think a person has the right to end his or her own life if this person has an incurable disease?", ['Yes', 'No'], [1, 2]),
 'pornlaw': ("Which of these statements comes closest to your feelings about pornography laws?", ['There should be laws against the distribution of pornography whatever the age', 'There should be laws against the distribution of pornography to persons under 18', 'There should be no laws forbidding the distribution of pornography'], [1, 2, 3]),
 'courts': ("In general, do you think the courts in this area deal too harshly or not harshly enough with criminals?", ['Too harshly', 'Not harshly enough', 'About right'], [1, 2, 3]),
 'happy': ("Taken all together, how would you say things are these days -- would you say that you are very happy, pretty happy, or not too happy?", ['Very happy', 'Pretty happy', 'Not too happy'], [1, 2, 3]),
 'satfin': ("So far as you and your family are concerned, would you say that you are pretty well satisfied with your present financial situation, more or less satisfied, or not satisfied at all?", ['Pretty well satisfied', 'More or less satisfied', 'Not satisfied at all'], [1, 2, 3]),
 'finrela': ("Compared with American families in general, would you say your family income is far below average, below average, average, above average, or far above average?", ['Far below average', 'Below average', 'Average', 'Above average', 'Far above average'], [1, 2, 3, 4, 5]),
 'getahead': ("Some people say that people get ahead by their own hard work; others say that lucky breaks or help from other people are more important. Which do you think is most important?", ['Hard work most important', 'Hard work and luck equally important', 'Luck or help from other people most important'], [1, 2, 3]),
}
# partition: (description for inverse prompt, [(group key, persona sentence, inverse option label)])
PARTS = {
 'party': ("political party identification", [('dem', 'Politically, you are a Democrat.', 'Democrat'), ('ind', 'Politically, you are an Independent.', 'Independent'), ('rep', 'Politically, you are a Republican.', 'Republican')]),
 'ideo': ("political ideology", [('lib', 'You consider yourself politically liberal.', 'Liberal'), ('mod', 'You consider yourself politically moderate.', 'Moderate'), ('con', 'You consider yourself politically conservative.', 'Conservative')]),
 'degree': ("education", [('nocoll', 'You do not have a four-year college degree.', 'No four-year college degree'), ('coll', "You have a bachelor's degree or higher.", "Bachelor's degree or higher")]),
 'age': ("age group", [('18-29', 'You are between 18 and 29 years old.', '18-29'), ('30-49', 'You are between 30 and 49 years old.', '30-49'), ('50-64', 'You are between 50 and 64 years old.', '50-64'), ('65+', 'You are 65 years old or older.', '65 or older')]),
 'sex': ("sex", [('male', 'You are a man.', 'Male'), ('female', 'You are a woman.', 'Female')]),
 'race': ("race", [('white', 'You are White.', 'White'), ('black', 'You are Black.', 'Black'), ('other', 'You are of another race (neither White nor Black).', 'Other race')]),
 'attend': ("religious service attendance", [('rare', 'You never or almost never attend religious services.', 'Never or less than once a year'), ('some', 'You attend religious services between a few times a year and a few times a month.', 'A few times a year to a few times a month'), ('weekly', 'You attend religious services nearly every week or more often.', 'Nearly every week or more')]),
}
L = 'ABCDEFG'
HEAD = "You are a respondent in the General Social Survey, a nationally representative survey of adults in the United States, around the year 2020."


def fwd(q, opts, sent=None):
    return (HEAD + (" " + sent if sent else "") + f"\n\n{q}\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer with a single letter.")


def inv(q, ans, desc, labels):
    a = f'\nThe respondent answered: "{ans}"' if ans is not None else ""
    return (f"In the General Social Survey (a nationally representative survey of US adults, around 2020), a respondent was asked:\n\"{q}\"{a}\n\n"
            f"What is this respondent's {desc}?\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(labels)) + "\nAnswer with a single letter.")


if __name__ == '__main__':
    P, O, K = [], [], []
    for it, (q, opts, _) in ITEMS.items():
        P.append(fwd(q, opts)); O.append(len(opts)); K.append(('fwd', it, 'pop', 'pop'))
        for pn, (desc, gs) in PARTS.items():
            for g, sent, _lab in gs:
                P.append(fwd(q, opts, sent)); O.append(len(opts)); K.append(('fwd', it, pn, g))
            labs = [x[2] for x in gs]
            P.append(inv(q, None, desc, labs)); O.append(len(labs)); K.append(('invprior', it, pn, ''))
            for j, o in enumerate(opts):
                P.append(inv(q, o, desc, labs)); O.append(len(labs)); K.append(('inv', it, pn, j))
    print('calls', len(P), 'max chars', max(map(len, P)), flush=True)
    t0 = time.time(); out = []
    for n in sorted(set(O)):
        idx = [i for i, o in enumerate(O) if o == n]
        res = score_options([P[i] for i in idx], list(L[:n]))
        for i, r in zip(idx, res):
            out.append(dict(kind=K[i][0], item=K[i][1], part=K[i][2], key=K[i][3], lp=[r[c] for c in L[:n]]))
        print(n, len(idx), round(time.time() - t0, 1), flush=True)
    json.dump(out, open((_REPO + '/outputs/screen/t4_llm.json'), 'w'))
    print('example fwd:\n', P[1], '\n\nexample inv:\n', P[5])
