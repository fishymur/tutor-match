# Tutor Matching Algorithm

**The blueprint for a matching algorithm that matches students with specefic subject needs to tutors who specialize in those requests.**

With the increasing use of AI in learning, the higher education has become seemingly more accessible. This, however, comes at a cost, a lack of structural learning that ignores proper prerequisite structuring and a reliance on biases that AI feeds to students who would have learned content in a much different enviroment otherwise. Generic tutoring platforms and educational platforms such as khan academy and coursera lack involved tutoring and real conversations between specialists and learners. The aim of this algorithm is to create a platform around tutor matching for students where not just categories and tags are matched for broad needs, but more specefic queries such as "why won't my neural net converge" can be processed and point to the best specialist to answer the question. The issue we aim to target is the jargon gap: students tend to describe their problems imprecisely and specifically, while tutors advertise expertise using broad, formal, and technical terminology. A keyword search fails to connect queries like “how to make my code run more efficiently” with “algorithmic complexity analysis,” even though they refer to the same concept. To bridge the semantic gap, this repository outlines a system that connects informal queries with formal tutor labels.

The surrounding design around reviews, bookings, payments, etc are deferred by design. The matching is the intent of the repo, later to be scaled

## Failure

We first observe the most simple case in which a query fails on generic educational platforms:

**Pure Keyword Baseline** - A query like "I need help with conditional expectation and Bayes theorum" scores 0 against a tutor who specializes in "probability theory, measure theory" since there are no directly shared words between the two.

**Semantic Matching Improvement** - The previous query now scores significantly higher since we are now using a embedding space where the query sits near the expertise in terms of meaning and context.

**Adding the Intake Agent** - A informal query that would have been matched poorly by semantics alone are rewritten initially by an agent, then embedded for improved accuracy.

## Framework

Student Request -> [Intake Agent (LLM rewrite claude-haiku-4-5) -> Embed -> Vector Search -> Hard Categorical Filtering -> Re-rank w/ Weighted Variables] -> Rankings with Categorical Explanation

1. **Intake Agent** - Claude Haiku 4.5 rewrites vague student query and condenses into precise topics and vocabulary used by tutors before embedding.
2. **Semantic Retrieval** - All tutors in database and query are all embedded into shared space; retrieval by cosine similarity.
3. **Filter** - Filter out by availibility of tutors and other filters put on by users.
4. **Re-rank** - Weighted balance of semantic relevance of query to tutor, ratings, price, etc. All categories normalized and weighted accordingly.
5. **Explanation** - Outputs match's component score breakdown.

## Evaluation

Match quality was determined using three measures. Recall1: correct tutors ranked first place. Recall3: correct tutors ranked in the top three. MMR: Mixture of recall 1 and recall3 that weighs higher ranking with more weight. The benchmark was 20 plain English queries phrased generically as a student would alongside tutors with purposefully overlapping expertise.

Using pure semantic retrieval and query, recall1 = 0.75, recall3 = 0.9, MRR = 0.845.

Using LLM intake agent: recall1 = 0.95, recall3 = 1, MRR = 0.967.

## Comparisons

In the case where a bigger embedding model did not help (all-MiniLM-L6-v2 vs all-mpnet-base-v2), the intake agent had a significant impact on the error value in the pure semantic retrieval case. Recall1 went up by 0.2, saving exactly the queries where a student's wording shared no surface terms with the correctly labled tutor.

## Design Choices

1. Normalization of variables before blending: signals for the tutor ranking are scaled 0-1 before turning it into a weighted sum so values such as price and rating can be accounted for.
2. Filtering out hard constraints: things like time-availibility misalignment, language differences, and filters put on by the users are explictly filtered out of the results.
3. Simple benchmark: The test set was small yet intentionally confusing with the use of overlapping tutors and messy phrasing so that we do not saturate a perfect score that cannot be improved.
4. Adaptive intake: LLM rewrites are only invoked when the top retrieval score is low, indicating low certainty; this pays for the calls only when retrieval looks vague.

## Tech Stack

Python · [`sentence-transformers`] (`all-mpnet-base-v2`) · Anthropic API (`claude-haiku-4-5` · custom Recall@k / MRR harness.
**Planned:** PostgreSQL + `pgvector`, a search API, Stripe, and a deployed frontend.

## Project Structure

## Local Running 

```bash
python3 -m pip install sentence-transformers anthropic
export ANTHROPIC_API_KEY= "sk-ant-..."   # read from env, never committed to source
 
python3 match_embeddings.py   # interactive matcher — type a request, get ranked matches
python3 evaluate.py           # run the benchmark, with and without the intake agent
```

## Limitation and Directions

