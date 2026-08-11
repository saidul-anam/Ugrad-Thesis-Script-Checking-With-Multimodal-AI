# Backend comparison report

Generated: 2026-08-11T11:17:22+00:00  
Backends: gemini, mistral_ocr

## Overview

| metric | gemini | mistral_ocr |
|---|---|---|
| model version | gemini-3.6-flash | mistral-ocr-latest |
| supports_prompt | True | False |
| pages transcribed | 34 | 34 |
| failed calls (incl. retries) | 1 | 0 |
| task units segmented | 10 | 10 |
| wall-clock (gemini) | SE_11_Q1_0001: 370s, SE_11_Q1_0002: 756s |
| wall-clock (mistral_ocr) | SE_11_Q1_0001: 36s, SE_11_Q1_0002: 47s |

## Fidelity markers per backend

A backend emitting zero markers on handwritten exam scripts is a signal,
not a success: it means unclear text was silently guessed or dropped.

| unit | gemini illegible/struck/cut | mistral_ocr illegible/struck/cut |
|---|---|---|
| CHART_001 | 0 / 5 / 0 | 0 / 0 / 0 |
| CHART_002 | 0 / 0 / 0 | 0 / 0 / 0 |
| LETTER_001 | 0 / 4 / 0 | 0 / 0 / 0 |
| LETTER_002 | 0 / 1 / 0 | 0 / 0 / 0 |
| PARA_001 | 0 / 4 / 0 | 0 / 0 / 0 |
| PARA_002 | 0 / 1 / 0 | 0 / 0 / 0 |
| STORY_001 | 0 / 5 / 0 | 0 / 0 / 0 |
| STORY_002 | 0 / 3 / 0 | 0 / 0 / 0 |
| SUMMARY_001 | 0 / 0 / 0 | 0 / 0 / 0 |
| SUMMARY_002 | 0 / 0 / 0 | 0 / 0 / 0 |
| **total** | **0 / 23 / 0** | **0 / 0 / 0** |

## Spelling-normalisation check (read-only)

Count of non-dictionary words per transcript (pyspellchecker, en). These
students make frequent spelling errors, so the backend with MORE
non-dictionary words is likely the more faithful one. Nothing is corrected.

| unit | gemini non-dict/total | mistral_ocr non-dict/total |
|---|---|---|
| CHART_001 | 6 / 120 | 5 / 125 |
| CHART_002 | 13 / 101 | 6 / 106 |
| LETTER_001 | 23 / 268 | 21 / 309 |
| LETTER_002 | 16 / 133 | 10 / 136 |
| PARA_001 | 6 / 198 | 3 / 201 |
| PARA_002 | 43 / 234 | 12 / 238 |
| STORY_001 | 4 / 162 | 5 / 167 |
| STORY_002 | 25 / 264 | 4 / 267 |
| SUMMARY_001 | 2 / 61 | 2 / 63 |
| SUMMARY_002 | 2 / 48 | 2 / 50 |
| **total** | **140 / 1589** | **70 / 1662** |

Non-dictionary words seen in **gemini**: 'a, 'ai', 'hope', aboud, accroding, adter, alaways, albenia, analysist, answear, beacuse, beggining, beller, charet, condrol, consumtion, coundry, creade, creadure, dake, danked, declaned, describ, dhad, dhan, dhanmondi, dhat, dhat's, dhe, dheir, dhey, diffenend, diny, disturcbed, dold, dool, droffic, eingineer, electricidy, enery, etc, fallfill, famaleys, featers, flys, fon, forc, gaibandha, graphe, greatful, hampen, hause, highesst, hsc, ii, iii, illustrades, impordand, indo, istory …

Non-dictionary words seen in **mistral_ocr**: 'ai', 'albenia', 'hope', afterpassing, beggining, camscanner, dhanmondi, etc, flys, graibandha, greatful, healthminiter, hsc, ii, iii, istory, jairif, jarit, jatrabara, jatrabari, jigatola, kalam, lanked, makin, nano, photoshop, rangpur, realised, saiful, salam, troppen, usa

## Red-ink / annotation leakage check

Heuristically flagged lines that look like examiner annotation or scanner
watermark. Over-flagging is intentional — read each against the scan.

**CHART_001 — gemini:**
- `the total sources. It was the`  ← examiner vocabulary

**CHART_001 — mistral_ocr:**
- `CS CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark

**CHART_002 — mistral_ocr:**
- `CS CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark

**LETTER_001 — mistral_ocr:**
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark

**LETTER_002 — mistral_ocr:**
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS CamScanner`  ← CamScanner watermark

**PARA_001 — mistral_ocr:**
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS CamScanner`  ← CamScanner watermark

**PARA_002 — mistral_ocr:**
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS CamScanner`  ← CamScanner watermark

**STORY_001 — mistral_ocr:**
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark

**STORY_002 — mistral_ocr:**
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark

**SUMMARY_001 — mistral_ocr:**
- `CS CamScanner`  ← CamScanner watermark

**SUMMARY_002 — mistral_ocr:**
- `CS`  ← CamScanner watermark
- `CamScanner`  ← CamScanner watermark

## Task units: side-by-side and word-level diff

### CHART_001

Word-level similarity: 0.878

<details><summary><b>gemini</b> full text</summary>

```text
The sources of the USA electricity
in 1980

In the given pie-charet we can see that
the sources of the USA electricity in
1980.

In 1980 The main source of generation
electricity in USA was [struck: 24%] [struck: Natural]
Coal [struck: gas] which was the highesst 46% of
the total sources. It was the
highest demand for generating electricity.
After that Natural gas the second
highest option for generating electricity
which was 24%. Then it was Hydro-
electric power which was 16% of total.
12% electricity a was generated from
oil. A Now the least used one was
Nuclear only 2%.

So we can make a [struck: col] conclusion that
they [struck: sued] used so much non-renewal
energy which are limited in nature. If
we want to save these natural resource
we should increase the use of renewable
enery.
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
# The sources of the USA electricity in 1980

In the given pie-chart we can see that the sources of the USA electricity in 1980.

In 1980 The main source of generation electricity in USA was Coal which was the highest of 46% of

CS CamScanner
the total sources. It was the highest demand for generating electricity. After that Natural gas the second highest option for generating electricity which was 24%. Then it was Hydro-electric power which was 16% of total. 12% electricity was generated from oil. Now the least used one was Nuclear only 2%. So, we can make a conclusion that they used so much non-renewable energy which are limited in nature. If we want to save these natural resources we should increase the use of renewable energy.

CS

CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

**[M only: #]** The sources of … In the given **[G: pie-charet | M: pie-chart]** we can see … in USA was **[G only: [struck: 24%] [struck: Natural]]** Coal **[G only: [struck: gas]]** which was the **[G: highesst | M: highest of]** 46% of **[M only: CS CamScanner]** the total sources. … Then it was **[G: Hydro- electric | M: Hydro-electric]** power which was … total. 12% electricity **[G only: a]** was generated from oil. **[G only: A]** Now the least … Nuclear only 2%. **[G: So | M: So,]** we can make a **[G only: [struck: col]]** conclusion that they **[G only: [struck: sued]]** used so much **[G: non-renewal | M: non-renewable]** energy which are … save these natural **[G: resource | M: resources]** we should increase the use of renewable **[G: enery. | M: energy. CS CamScanner]**

### CHART_002

Word-level similarity: 0.828

<details><summary><b>gemini</b> full text</summary>

```text
The pie chart illustrates the sources of
USA electricity in 1980.
Accroding do the chart, we can see that in 1980
most percentage of electricity in USA
comes from burning coal. which rate was
46% , nearly half of the chart. Natural
gas consumtion gives second highest
source for produce electricity, which is
a. 24%, Moreover, hydro electric power
plant produce 16% percent electricity
in USA. Oil and Nuclear power
plands also a impordand secton to
supply electricity . which rate is
respectively 12% and 2%
From dhe chart, we explore thab coal is dhe most
powerful sector in USA electricidy supply. Natural gas
and Hydro-electric power also, play a huge
role:
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
The pie chart illustrates the sources of USA electricity in 1980.

According to the chart, we can see that in 1980 most percentage of electricity in USA comes from burning coal, which rate was 46%, nearly half of the chart. Natural gas consumption gives second highest source for produce electricity, which is a 24%. Moreover, hydro electric power plant produce 16% percent electricity in USA. Oil and Nuclear power plants also a important sector to supply electricity, which rate is

CS CamScanner
respectively 12% and 2%

From the chart, we explore that coal is the most powerful sector in USA electricity supply. Natural gas and Hydro-electric power also play a huge role:

role:

CS

CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

The pie chart … electricity in 1980. **[G: Accroding do | M: According to]** the chart, we … comes from burning **[G: coal. | M: coal,]** which rate was **[G: 46% , | M: 46%,]** nearly half of the chart. Natural gas **[G: consumtion | M: consumption]** gives second highest … electricity, which is **[G: a. 24%, | M: a 24%.]** Moreover, hydro electric … and Nuclear power **[G: plands | M: plants]** also a **[G: impordand secton | M: important sector]** to supply **[G: electricity . | M: electricity,]** which rate is **[M only: CS CamScanner]** respectively 12% and 2% From **[G: dhe | M: the]** chart, we explore **[G: thab | M: that]** coal is **[G: dhe | M: the]** most powerful sector in USA **[G: electricidy | M: electricity]** supply. Natural gas and Hydro-electric power **[G: also, | M: also]** play a huge role: **[M only: role: CS CamScanner]**

### LETTER_001

Word-level similarity: 0.815

<details><summary><b>gemini</b> full text</summary>

```text
Jatrabari, Dhaka-1204
27-06-2026
Dear Saiful,
At the beggining of the letter take my salam
and the best wishes from the core of my heart.
In your letter you wanted to know
about my future plan after passing
the Hsc Exam. Today I'm writing so.
After my Hsc Exam I will get a lot of
free time which I am thinking to use
forc freelancing. Freelancing means working
from home. There are so many works
which can be done in freelancing. As such
as video editing, graphe & graphic design,
AI work and many more othere
works : It can be time consuming to
grow in this sector. Moreover, I can earn
money by doing this which can help me
to become self-relient.
No more today. You can also share what you
will do after the HSC Exam. Hoping to
hear from you soon.
You're's ever,
Kalam

From
To
Stamp
Kalam
Saiful
Jatrabari
Gaibandha
Dhaka-1204
Rangpur
Ans of the Question Number - 1 (A)

a. iii. being displaced lives outside the
country.
b. ii. flying objects.
c. iii. blockade.
d. i. rushed in large number.
e. i. coward.

Ans of the Question Number - 1 (B)

a. ans: The writer can't [struck: q] write beautiful
words about Gaza because he only saw
poverty, siege and famine.
b. ans: The writer only saw deprivation
in every hause, fear, sickness and
poverty and he [struck: became] became aware
of it.
c. The sea is the only thing [struck: wich] which
helps the author to go around the world.

d. ans: The author go back to the rea reality
of Gaza. d. and he looks for nice words
to say but he can't find them.

e. ans: The flies remind the author about
fighter planes which [struck: makes] made him scared of
any flying thing.
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
Jatrabari, Dhaka-1204

27-06-2026

Dear Saiful,

At the beggining of the letter take my salam
and the best wishes from the core of my heart.

In your letter you wanted to know
about my future plan after I passing

the HSC Exam. Today I am writing so.

After my HSC Exam I will get a lot of

free time which I am thinking to use

for freelancing. Freelancing means working

from home. There are so many works

which can be done in Freelancing. A such

as video editing, graphic design,

AI work and many more other

CS

CamScanner
works. It can be time consuming to grow in this sector. Moreover, I can earn money by doing this which can help me to become self-reliant.

No more today. You can also share what you will do after the HSC Exam. Hoping to hear from you soon.

Yours ever,

Kalam

|  From | To | Stamp  |
| --- | --- | --- |
|  Kalam | Saiful |   |
|  Jatrabara | Graibandha |   |
|  Dhaka-1804 | Rangpur |   |

CS

CamScanner
# Ans of the Question Number-1(A)

a. iii. being displaced lives outside the country.
b. ii. flying objects.
c. iii. blockade.
d. i. rushed in large number.
e. i. coward.

# Ans of the Question Number-1(B)

a. ans: The writer can't write beautiful words about Gaza because he only saw poverty, siege and famine.
b. ans: The writer only saw deprivation in every house, fear, sickness and poverty and he became aware of it.

CS

CamScanner
c. The sea is the only thing which helps the author to go around the world.

d. ans: The author go back to the reality of Gaza. and he looks for nice words to say but he can't find them.

e. ans: The flies remind the author about fighter planes which made him scared of

any flying thing.

CS

CamScanner
# Ans! Answer of the Question Number-2

Dropping out of school

working in in-law's household

facing dowry-related violence

increasing health risks

lacking information about reproductive health and contraception

losing their mobility

Cap Hall

CS

CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

Jatrabari, Dhaka-1204 27-06-2026 … future plan after **[M only: I]** passing the **[G: Hsc | M: HSC]** Exam. Today **[G: I'm | M: I am]** writing so. After my **[G: Hsc | M: HSC]** Exam I will … thinking to use **[G: forc | M: for]** freelancing. Freelancing means … be done in **[G: freelancing. As | M: Freelancing. A]** such as video editing, **[G only: graphe &]** graphic design, AI work and many more **[G: othere works : | M: other CS CamScanner works.]** It can be … me to become **[G: self-relient. | M: self-reliant.]** No more today. … from you soon. **[G: You're's | M: Yours]** ever, Kalam **[M only: |]** From **[M only: |]** To **[M only: |]** Stamp **[M only: | | --- | --- | --- | |]** Kalam **[M only: |]** Saiful **[G: Jatrabari Gaibandha Dhaka-1204 | M: | | | Jatrabara | Graibandha | | | Dhaka-1804 |]** Rangpur **[M only: | | CS CamScanner #]** Ans of the Question **[G: Number - 1 (A) | M: Number-1(A)]** a. iii. being … e. i. coward. **[M only: #]** Ans of the Question **[G: Number - 1 (B) | M: Number-1(B)]** a. ans: The writer can't **[G only: [struck: q]]** write beautiful words … deprivation in every **[G: hause, | M: house,]** fear, sickness and poverty and he **[G only: [struck: became]]** became aware of it. **[M only: CS CamScanner]** c. The sea is the only thing **[G only: [struck: wich]]** which helps the … back to the **[G only: rea]** reality of Gaza. **[G only: d.]** and he looks … fighter planes which **[G only: [struck: makes]]** made him scared of any flying thing. **[M only: CS CamScanner # Ans! Answer of the Question Number-2 Dropping out of school working in in-law's household facing dowry-related violence increasing health risks lacking information about reproductive health and contraception losing their mobility Cap Hall CS CamScanner]**

### LETTER_002

Word-level similarity: 0.799

<details><summary><b>gemini</b> full text</summary>

```text
27 June 2026
Faridabad,
Dhaka-1204

Dear Jarif,

Hope you are well by the grace of Almighty
In your last letter, you wanted to know about
my future plan after passing H.S.C. Now I describ
about it. You know H.S.C is the most prominent
exam in our country. So In our country. after
H.S.C all students [struck: t] will take Public University
admission preparation. My dream is after passing
hsc exam I will take hardest preparadion
for my University admission. You know thad,
from my childhood, I want to be a
eingineer. and admitted into a Dhaka University
So, thad's time is dhe most impordand
momenterm to fallfill my dream.
I cannot sday in day dreaming. Instead. I
want to give my best for my own dream.
No more today. Take care yourself and your
famaleys.

Yours ever
Mahin

From:
Mahin
Faridabad,
Dhaka, 1204

To,
STAMP
Jarif
Jigatola
Dhanmondi-18
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
27 June 2026

Faridabad,

Dhaka-1204

Dear Jarit,

Hope you are well by the grace of Almighty

In your last letter, you wanted to know about

my future plan after passing H.S.C. Now I described

about it. You know H.S.C is the most prominent

exam in our country, so in our country, after

H.S.C all students & will take Public University

admission preparation. My dream is afterpassing

hsc exam I will take hardest preparation

for my University admission. You know that

from my childhood, I want to be a

engineer and admitted into a Dhaka University

so that's time is the most important

momentum to fulfill my dream.

CS

CamScanner
I cannot stay in day dreaming. Instead, I
want to give my best for my own dream.
No more today, take care yourself and your
families.

Yours ever

Makin

From,

Makin

Faridabad,

Dhaka, 1204

To,

Jairif

Jigatola

Dhanmondi-18

STAMP

CS CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

27 June 2026 Faridabad, Dhaka-1204 Dear **[G: Jarif, | M: Jarit,]** Hope you are … H.S.C. Now I **[G: describ | M: described]** about it. You … exam in our **[G: country. So In | M: country, so in]** our **[G: country. | M: country,]** after H.S.C all students **[G: [struck: t] | M: &]** will take Public … My dream is **[G: after passing | M: afterpassing]** hsc exam I will take hardest **[G: preparadion | M: preparation]** for my University admission. You know **[G: thad, | M: that]** from my childhood, … to be a **[G: eingineer. | M: engineer]** and admitted into a Dhaka University **[G: So, thad's | M: so that's]** time is **[G: dhe | M: the]** most **[G: impordand momenterm | M: important momentum]** to **[G: fallfill | M: fulfill]** my dream. **[M only: CS CamScanner]** I cannot **[G: sday | M: stay]** in day dreaming. **[G: Instead. | M: Instead,]** I want to … dream. No more **[G: today. Take | M: today, take]** care yourself and your **[G: famaleys. | M: families.]** Yours ever **[G: Mahin From: Mahin | M: Makin From, Makin]** Faridabad, Dhaka, 1204 To, **[G: STAMP Jarif | M: Jairif]** Jigatola Dhanmondi-18 **[M only: STAMP CS CamScanner]**

### PARA_001

Word-level similarity: 0.918

<details><summary><b>gemini</b> full text</summary>

```text
Artificial Intelligence

Artificial Intelligence mostly known as AI
is a modern technology which contains
hardware and software and try to do
tasks like human brain. It is nowadays
used widely in all sectors like teaching,
nursing, training etc. Doctors can use
AI to define any problem presicely. AI
also can help students at making notes.
[struck: Free] Freelancers use AI. to do their work
fast, which saves both time, and enery.
So, AI is seen in all sectors and it
is increasing rapidly because of it's
usefulness. [struck: Which] Some task are time consuming
and hard AI can easily solve them
at no time. It [struck: a] has become an
important part of our daily life.
Beacuse of AI teaching and learning
got revolutionized. It can help teacher
to make gather information about
any topic it can also make a slide
which can be used for teaching students.
AI is very useful for us but it also
have some threatening aspects such as
taking peples job, data loss, privecy lack
and It has become so powerful that
if anyone uses it for any bad
work like hacking, it can affect the
whole m nation. So, we all should use
this modern technology for the pleasure
of [struck: manking] mankind.
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
## Artificial Intelligence

Artificial Intelligence mostly known as AI is a modern technology which contains hardware and software and try to do tasks like human brain. It is nowadays used widely in all sectors like teaching, nursing, training etc. Doctors can use AI to define any problem precisely. AI also can help students at making notes. Freelancers use AI to do their work fast, which saves both time, and energy. So, AI is seen in all sectors and it is increasing rapidly because of it's

CS

CamScanner
usefulness. Some task are time consuming and hard AI can easily solve them at no time. It has become an important part of our daily life. Because of AI teaching and learning got revolutionized. It can help teachers to gather information about any topic it can also make a slide which can be used for teaching students. AI is very useful for us but it also have some threatening aspects such as taking peoples job, data loss, privacy, lack and it has become so powerful that if anyone uses it for any bad

CS CamScanner
3 work like hacking it can affect the whole nation. So, we all should use this modern technology for the pleasure of mankind.
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

**[M only: ##]** Artificial Intelligence Artificial … define any problem **[G: presicely. | M: precisely.]** AI also can … at making notes. **[G only: [struck: Free]]** Freelancers use **[G: AI. | M: AI]** to do their … both time, and **[G: enery. | M: energy.]** So, AI is … because of it's **[M only: CS CamScanner]** usefulness. **[G only: [struck: Which]]** Some task are … no time. It **[G only: [struck: a]]** has become an … our daily life. **[G: Beacuse | M: Because]** of AI teaching … It can help **[G: teacher | M: teachers]** to **[G only: make]** gather information about … such as taking **[G: peples | M: peoples]** job, data loss, **[G: privecy | M: privacy,]** lack and **[G: It | M: it]** has become so … for any bad **[M only: CS CamScanner 3]** work like **[G: hacking, | M: hacking]** it can affect the whole **[G only: m]** nation. So, we … the pleasure of **[G only: [struck: manking]]** mankind.

### PARA_002

Word-level similarity: 0.760

<details><summary><b>gemini</b> full text</summary>

```text
Artificial Intelligence

Artificial Intelligence is called 'AI'
which means one acts like a human brain.
when we ask any question it gives us
correctly answear. It works like human
brain and recovery their wrong answear.
Nowdays, AI have many featers. Like
answear and question, deep learning, Nano
banana, instant video maker, analysis, create
istory, photoshop dool etc. For instant moment,
we take knowledge from 'AI'. If we
want to write new thoughts it's
helps us to know about the world.
Around the world, every coundry, nation
use AI in dheir daily activities. From
morning too dill night we use it. when
indo a m special moment and it remind
me in time. Moreover. Now days government
use it in Dhaka droffic systeme and we
can dake dhe advantage. and our
droffic condrol system must be beller dhan
past. In dhe world. we see dhad.
Albenia declaned first AI health miniten
and it is dhe first in world whene
worked as a ministen. Moreover we
use it, diffenend sector. AI has
creade a revolutionized teaching
and loanning. we can ask
any question in AI and set dhe reply
in a mument. It's also work a research
analysist. dhey can know all things with
AI. It's work like a libary. a where we
can get dhe [struck: St.] knowledge. we find dhat
one day 'AI' work in factory. mills and all
section. It also hampen our human activity
with wont AI we cannot stay in deep
thinking.
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
# "Artificial Intelligence"

Artificial Intelligence is called 'AI'

which acts like a human brain.

when we ask any question it gives us

correctly answer. It works like human

brain and recovery their wrong answer.

Nowadays, AI have many features. Like

answer and question, deep learning, Nano

banana, instant video maker, analysis, create

istory, photoshop tool etc. For instant moment,

we take knowledge from 'AI'. If we

want to write new thoughts it's

helps us to know about the world.

CS

CamScanner
Around the world, every country, nation

age AI in their daily activities. From

morning too till night we use it when set

into a special moment and it remind

me in time. Moreover, Nowadays government

use it in Dhaka traffic systems and we

can take the advantage and our

traffic control system must be better than

past. In the world, we see that

'Albenia' declared first AI healthminiter

and it is the first in world where

worked as a minister. Moreover we

use it different sector. AI has

create a revolutionized teaching

and learning. we can ask

CS

CamScanner
JUN 2023

any question in AI and set the reply in a moment. It's also work a research analysis. They can know all things with AI. It's work like a library, where we can get the knowledge, we find that one day 'AI' work in factory, mills and all sector. It also hampers our human activity with work AI we cannot stay in deep thinking.

CS CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

**[G: Artificial Intelligence | M: # "Artificial Intelligence"]** Artificial Intelligence is called 'AI' which **[G only: means one]** acts like a … gives us correctly **[G: answear. | M: answer.]** It works like … recovery their wrong **[G: answear. Nowdays, | M: answer. Nowadays,]** AI have many **[G: featers. | M: features.]** Like **[G: answear | M: answer]** and question, deep … create istory, photoshop **[G: dool | M: tool]** etc. For instant … about the world. **[M only: CS CamScanner]** Around the world, every **[G: coundry, | M: country,]** nation **[G: use | M: age]** AI in **[G: dheir | M: their]** daily activities. From morning too **[G: dill | M: till]** night we use **[G: it. | M: it]** when **[G: indo | M: set into]** a **[G only: m]** special moment and … me in time. **[G: Moreover. Now days | M: Moreover, Nowadays]** government use it in Dhaka **[G: droffic systeme | M: traffic systems]** and we can **[G: dake dhe advantage. | M: take the advantage]** and our **[G: droffic condrol | M: traffic control]** system must be **[G: beller dhan | M: better than]** past. In **[G: dhe world. | M: the world,]** we see **[G: dhad. Albenia declaned | M: that 'Albenia' declared]** first AI **[G: health miniten | M: healthminiter]** and it is **[G: dhe | M: the]** first in world **[G: whene | M: where]** worked as a **[G: ministen. | M: minister.]** Moreover we use **[G: it, diffenend | M: it different]** sector. AI has **[G: creade | M: create]** a revolutionized teaching and **[G: loanning. | M: learning.]** we can ask **[M only: CS CamScanner JUN 2023]** any question in AI and set **[G: dhe | M: the]** reply in a **[G: mument. | M: moment.]** It's also work a research **[G: analysist. dhey | M: analysis. They]** can know all … work like a **[G: libary. a | M: library,]** where we can get **[G: dhe [struck: St.] knowledge. | M: the knowledge,]** we find **[G: dhat | M: that]** one day 'AI' work in **[G: factory. | M: factory,]** mills and all **[G: section. | M: sector.]** It also **[G: hampen | M: hampers]** our human activity with **[G: wont | M: work]** AI we cannot stay in deep thinking. **[M only: CS CamScanner]**

### STORY_001

Word-level similarity: 0.904

<details><summary><b>gemini</b> full text</summary>

```text
Size doesn't Matter

Once a lion was sleeping in a forrest.
Suddenly a mouse came there. It was,
in fact, playing on a race quite happily
and didn't notice the sleeping lion. While
running, it's [struck: a] sound woke up the lion.
The lion caught the mouse. The mouse
got scared and told the lion to an to
let him free, one day he might ka save
the lion's life. Hearing this the lion
laughed and told the mouse that he
was small how even can he save the
life of the lion. At last the lion let
[struck: Freed] the mouse go. One day, in
the jungle the mouse heard a sound
he followed the sound and saw
that the lion was trappen in a net
which was put there by the hunters.
He went to the lion and cut the net
with his sharp [struck: teath] teeth . The lion
was greatful for him as the mouse
[struck: have] saved his life .. Then he re
[struck: the] the lion realised size doesn't matter
to do any worck.
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
# Size doesn't Matter

Once a lion was sleeping in a forest.

Suddenly a mouse came there. It was,

in fact, playing on a race quite happily

and didn't notice the sleeping lion. While

running, it's sound woke up the lion.

The lion caught the mouse. The mouse

got scared and told the lion to go to

let him free, one day he might have save

the lion's life. Hearing this the lion

laughed and told the mouse that he

was small how even can he save the

life of the lion. At last the lion let

free the mouse go. One day, in

CS

CamScanner
the jungle the mouse heard a sound

he followed the sound and saw

that the lion was troppen in a net

which was put there by the hunters.

He went to the lion and cut the net

with his sharp teeth. The lion

was greatful for him as the mouse

have saved his life. Then he

the lion realised size doesn't matter

to do any work.

CS

CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

**[M only: #]** Size doesn't Matter … sleeping in a **[G: forrest. | M: forest.]** Suddenly a mouse … While running, it's **[G only: [struck: a]]** sound woke up … the lion to **[G: an | M: go]** to let him … day he might **[G: ka | M: have]** save the lion's … the lion let **[G: [struck: Freed] | M: free]** the mouse go. One day, in **[M only: CS CamScanner]** the jungle the … the lion was **[G: trappen | M: troppen]** in a net … with his sharp **[G: [struck: teath] teeth . | M: teeth.]** The lion was … as the mouse **[G: [struck: have] | M: have]** saved his **[G: life .. | M: life.]** Then he **[G only: re [struck: the]]** the lion realised … to do any **[G: worck. | M: work. CS CamScanner]**

### STORY_002

Word-level similarity: 0.730

<details><summary><b>gemini</b> full text</summary>

```text
" [struck: An] A. Lion and [struck: a] Mouse "

Once a lion was sleeping in a forest.
Suddenly a mouse came there. It was
in fact, playing on a race quite happily
and didn't notice the sleeping lion. while
running into his body, he entered the Lion
nose, and this time the lion wake up.
with his roaring, he told him " How
dare youa a tiny creature." You don't know
me. why you disturcbed me. Now I will
kill you"? The mouse reply the faint voic
thad "don't kill me!, Lord; I cannot
see you. I only play outside you,
I am a diny creadure , if you killed me
dhat's not fame fort Lord." The Lion
reply dhat " you disturb ' me in my sleep.
as a diny creadure. You cannot sive me
proper respect " The mouse reply dhat
" My Lord, please Forgive me, Oneday I will
help you ." The Lion laughed and dold dhat
"You, a diny creadure helped me. Now, I
forgive you, but ad the next dime be
careful aboud myself. so away." The mouse
danked him and go fan away. On adter
passing some days . dhe mouse searching
fon food. in dhe forrest, suddenly, he
hear something. Then [struck: t] the mouse
cleanly hear that 'A lion was roaring'
Then, he so to the place and see
some unpredicdable. The mouse saw the
lion and remember the lion who forgive
him captured in and The mouse told faint voice " Friend!
Don't be afraid. I will help. you. as I had
he so do the net. and cut the net
with his sharp teeth. Then the lion
be free. The - lion told that " Thank you
my friend. for saving my life. from today
we are friend"
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
C. A. Lion and C. Mouse

Once a lion was sleeping in a forest.

Suddenly a mouse came there. It was

in fact, playing on a race quite happily

and didn't notice the sleeping lion. while

running into his body, he entered the lion

nose, and this time the lion wake up.

with his roaring, he told him "How

dare you a tiny creature," You don't know

me. why you disturbed me. Now I will

tell you." The mouse reply the faint voice

that "don't kill me! Lord; I cannot

see you. I only play outside you,

CS

CamScanner
I am a tiny creature, if you killed me
that's not fame for Lord." The Lion
reply that "you disturb me in my sleep.
as a tiny creature. you cannot give me
proper respect." The mouse reply that
"My Lord, please forgive me. One day I will
help you." The Lion laughed and told that
"You, a tiny creature helped me. Now, I
forgive you, but at the next time be
careful about myself, so away." The mouse
lanked him and go far away. an other
passing some days. the mouse searching
for food. in the forest. suddenly. he

CS

CamScanner
hear something. Then the mouse clearly hear that "A lion was roaring" Then, he go to the place and see some unpredictable. The mouse saw the lion and remember the lion who forgive him. The mouse told faint voice "Friend! Don't be afraid, I will help you, at that he go to the net, and cut the net with his sharp teeth. Then the lion be free. The lion told that "Thank you my friend for saving my life from today we are friend"

CS

CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

**[G: " [struck: An] | M: C.]** A. Lion and **[G: [struck: a] | M: C.]** Mouse **[G only: "]** Once a lion … he entered the **[G: Lion | M: lion]** nose, and this … he told him **[G: " How | M: "How]** dare **[G: youa | M: you]** a tiny **[G: creature." | M: creature,"]** You don't know me. why you **[G: disturcbed | M: disturbed]** me. Now I will **[G: kill you"? | M: tell you."]** The mouse reply the faint **[G: voic thad | M: voice that]** "don't kill **[G: me!, | M: me!]** Lord; I cannot … play outside you, **[M only: CS CamScanner]** I am a **[G: diny creadure , | M: tiny creature,]** if you killed me **[G: dhat's | M: that's]** not fame **[G: fort | M: for]** Lord." The Lion reply **[G: dhat " you | M: that "you]** disturb **[G only: ']** me in my sleep. as a **[G: diny creadure. You | M: tiny creature. you]** cannot **[G: sive | M: give]** me proper **[G: respect " | M: respect."]** The mouse reply **[G: dhat " My | M: that "My]** Lord, please **[G: Forgive me, Oneday | M: forgive me. One day]** I will help **[G: you ." | M: you."]** The Lion laughed and **[G: dold dhat | M: told that]** "You, a **[G: diny creadure | M: tiny creature]** helped me. Now, I forgive you, but **[G: ad | M: at]** the next **[G: dime | M: time]** be careful **[G: aboud myself. | M: about myself,]** so away." The mouse **[G: danked | M: lanked]** him and go **[G: fan | M: far]** away. **[G: On adter | M: an other]** passing some **[G: days . dhe | M: days. the]** mouse searching **[G: fon | M: for]** food. in **[G: dhe forrest, suddenly, | M: the forest. suddenly.]** he **[M only: CS CamScanner]** hear something. Then **[G only: [struck: t]]** the mouse **[G: cleanly | M: clearly]** hear that **[G: 'A | M: "A]** lion was **[G: roaring' | M: roaring"]** Then, he **[G: so | M: go]** to the place and see some **[G: unpredicdable. | M: unpredictable.]** The mouse saw … lion who forgive **[G: him captured in and | M: him.]** The mouse told faint voice **[G: " Friend! | M: "Friend!]** Don't be **[G: afraid. | M: afraid,]** I will **[G: help. you. as I had | M: help you, at that]** he **[G: so do | M: go to]** the **[G: net. | M: net,]** and cut the … be free. The **[G only: -]** lion told that **[G: " Thank | M: "Thank]** you my **[G: friend. | M: friend]** for saving my **[G: life. | M: life]** from today we are friend" **[M only: CS CamScanner]**

### SUMMARY_001

Word-level similarity: 0.968

<details><summary><b>gemini</b> full text</summary>

```text
In the poem Hope is compared with
bird. Bird alaways flys here and
there. In the jungle in rain, in
winter and also in on the strangest
sea we can hear it. It's singing
keeps many people warm. Like the
bird we can find hope anywhere
in any situation. In our best times
or on the times where we are
alone.
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
In the poem Hope is compared with bird. Bird always flys here and there. In the jungle in rain, in winter and also in on the strangest sea we can hear it. It's singing keeps many people warm. Like the bird we can find hope anywhere in any situation. In our best times or on the times where we are alone.

CS CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

In the poem … with bird. Bird **[G: alaways | M: always]** flys here and … we are alone. **[M only: CS CamScanner]**

### SUMMARY_002

Word-level similarity: 0.898

<details><summary><b>gemini</b> full text</summary>

```text
The poem 'Hope' illustrades Hope lives in the
soul and it sings endlessly at the darkest moment
in the life. It never ask anything return. In the
storm of life it gives us strength and courage
Moreover, in the extreme situation of the hope
give as mental comfort.
```
</details>

<details><summary><b>mistral_ocr</b> full text</summary>

```text
The poem 'Hope' illustrates Hope lives in the soul and it sings endlessly at the darkest moment in the life. It never ask anything return. In the storm of life it gives us strength and courage. Moreover, in the extreme situation of the hope she is mental comfort.

CS

CamScanner
```
</details>

**Word diff (G = gemini, M = mistral_ocr):**

The poem 'Hope' **[G: illustrades | M: illustrates]** Hope lives in … us strength and **[G: courage | M: courage.]** Moreover, in the … of the hope **[G: give as | M: she is]** mental comfort. **[M only: CS CamScanner]**

## Cost, tokens, and projection

| | gemini | mistral_ocr |
|---|---|---|
| pilot usage | {'input': 57052, 'output_incl_thinking': 151630} | {'pages_processed': 34} |
| pilot cost | $1.2228 | $0.1360 |
| projected cost, 200 scripts | $122.28 | $13.60 |
| avg wall-clock / script | 563s | 42s |

> Pricing used (2026-08-11): Gemini 3.6 Flash $1.50/M input, $7.50/M output,
> thinking tokens billed at the output rate (batch API is half);
> Mistral OCR 4 $4.00/1000 pages (synchronous; batch API is half).
