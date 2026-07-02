"""
generate_sample_data.py
------------------------
Creates a small, offline sample dataset (sample_toxicity_data.csv) so that
text_toxicity_moderation.ipynb can run end-to-end WITHOUT downloading the
full Jigsaw / Civil Comments dataset from Kaggle.

The full real-world dataset can be downloaded separately -- see data/README.md.

This sample uses mild, non-graphic examples of toxic/insulting language
(no slurs, no hate speech targeting protected groups) purely for
demonstrating the moderation pipeline in an educational assignment.
"""

import csv
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Non-toxic (label = 0) example comments -- polite / neutral / constructive
# ---------------------------------------------------------------------------
non_toxic = [
    "Thank you so much for explaining this, it really helped me understand.",
    "I disagree with your point, but I appreciate the detailed explanation.",
    "Great article! Looking forward to reading more from you.",
    "Could you please share the source for this claim?",
    "This is a well-written and balanced summary of the issue.",
    "I had a similar experience, thanks for sharing yours.",
    "Let's keep the discussion respectful and focus on the facts.",
    "The customer support team resolved my issue quickly and politely.",
    "I love how the community here is so supportive and helpful.",
    "Can someone recommend a good beginner tutorial for this topic?",
    "The weather has been lovely this week, perfect for a walk.",
    "I think we should schedule a follow-up meeting to discuss next steps.",
    "Congratulations on your promotion, you truly earned it!",
    "The new update fixed the bug I reported last week, thank you.",
    "I'm not sure I agree, could you elaborate a bit more?",
    "This recipe turned out delicious, thanks for posting it.",
    "Please let me know if you need any more information from my side.",
    "The movie was okay, not the best but still enjoyable.",
    "I appreciate your patience while we sort out this issue.",
    "Looking forward to collaborating with the team on this project.",
    "Your presentation was clear and easy to follow, well done.",
    "I think there might be a small typo in paragraph two.",
    "Happy to help if you have any more questions about the setup.",
    "The product arrived on time and works exactly as described.",
    "It was nice meeting everyone at the conference yesterday.",
    "I respectfully disagree, here is some data that supports my view.",
    "Thanks for the quick reply, that clears things up.",
    "Our team achieved the quarterly target, great job everyone.",
    "The documentation could be improved with a few more examples.",
    "I really admire how you handled that difficult situation calmly.",
    "Could we reschedule the call to tomorrow afternoon instead?",
    "The bug fix works perfectly now, thanks for the quick turnaround.",
    "I found this tutorial very clear and beginner friendly.",
    "Let's brainstorm some ideas for the next sprint planning session.",
    "The food at the new restaurant downtown is amazing.",
    "I appreciate the feedback, I will make those changes soon.",
    "That's an interesting perspective, I hadn't thought of it that way.",
    "The training session was informative and well organized.",
    "Please review the attached document and share your comments.",
    "Wishing you a speedy recovery, take care of yourself.",
    "The library has a great collection of programming books.",
    "I think the design looks clean and modern, nice work.",
    "Can you walk me through the steps to reproduce this issue?",
    "The support article answered all my questions, thank you.",
    "It's always a pleasure working with such a dedicated team.",
    "I updated the ticket with the latest logs, please check.",
    "Thanks for volunteering to organize the event this year.",
    "The onboarding process was smooth and well documented.",
    "I appreciate you taking the time to review my code.",
    "Have a great weekend, see everyone on Monday!",
]

# ---------------------------------------------------------------------------
# Toxic (label = 1) example comments -- insults, harassment, profanity.
# Kept mild/generic on purpose: no slurs, no hate speech vs protected groups.
# Good enough to teach a classifier the general shape of toxic language.
# ---------------------------------------------------------------------------
toxic = [
    "You are such an idiot, how do you not understand this simple thing.",
    "Shut up already, nobody cares about your stupid opinion.",
    "This is the dumbest article I have ever read, the writer is a moron.",
    "Get lost, you clueless loser, you're wasting everyone's time.",
    "You're a pathetic excuse for a developer, this code is garbage.",
    "Stop being so annoying, you absolute idiot.",
    "What a worthless piece of trash this product is, and so is the seller.",
    "You people are all brainless idiots who can't think for yourselves.",
    "I hope you fail, you disgusting waste of space.",
    "Only a complete moron would post something this stupid.",
    "You are a joke, nobody respects you, just quit already.",
    "This is garbage, you're garbage, everything about this is garbage.",
    "Go away, you pathetic little troll, nobody wants you here.",
    "You are so stupid it actually hurts to read your comment.",
    "Everyone thinks you're an incompetent fool, just admit it.",
    "Shut your mouth, you ignorant fool, you know nothing.",
    "You're a disgrace and an embarrassment to this community.",
    "What an absolute idiot, I can't believe how dumb this take is.",
    "You are worthless and everyone here despises you.",
    "Delete your account, you brainless clown, you add nothing of value.",
    "This is trash content from a talentless hack.",
    "You are the most annoying, useless person on this forum.",
    "I can't stand how stupid and arrogant you are, just stop talking.",
    "Nobody likes you, you pathetic attention seeker.",
    "You're an idiot and everything you say is nonsense.",
    "Get out of here with your garbage opinions, loser.",
    "You clearly have no brain, this comment proves it.",
    "This whole team is a bunch of incompetent losers.",
    "You are disgusting and everyone should ignore your trash takes.",
    "What a moronic thing to say, are you always this dumb?",
    "You're a fraud and a liar, nobody should trust a word you say.",
    "I hope you get fired, you lazy useless excuse for an employee.",
    "You absolute clown, learn to code before embarrassing yourself.",
    "Your opinion is trash and so are you, just shut up.",
    "Everyone here thinks you're a pathetic, whiny little baby.",
    "You are a talentless hack who should never be allowed to write again.",
    "Stop spamming this forum with your idiotic nonsense.",
    "You're an embarrassment to your family, honestly pathetic.",
    "This is the worst take I've ever seen from a total imbecile.",
    "Nobody asked for your garbage opinion, keep it to yourself, loser.",
    "You are so incredibly dumb, it's a wonder you can even type.",
    "Get a life, you sad, pathetic troll.",
    "You're a coward and a liar, disgusting behavior honestly.",
    "This app is garbage and whoever built it is a talentless idiot.",
    "You're an absolute waste of oxygen, stop posting.",
    "Only an idiot would defend something this stupid.",
    "You're a disgusting human being and everyone can see it.",
    "Shut up, moron, your comment adds zero value here.",
    "You are pathetic, weak, and honestly just an idiot.",
    "I despise how arrogant and stupid you sound right now.",
]

rows = [(t, 1) for t in toxic] + [(t, 0) for t in non_toxic]
random.shuffle(rows)

with open("sample_toxicity_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["comment_text", "toxic"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to sample_toxicity_data.csv "
      f"({len(toxic)} toxic, {len(non_toxic)} non-toxic)")
