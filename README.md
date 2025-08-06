# An Interest-Based Cardinal Social Choice System

This project designs, simulates, and analyzes a novel voting framework aimed at addressing fundamental flaws in traditional voting systems, such as issue-bundling and the suppression of minority interests. The system allows voters to express both their preference on an issue and the *intensity* of that preference by allocating a fixed budget of "voting power" across multiple elections.

### Core Concepts

Traditional voting systems often force voters into false dichotomies (e.g., bundling economic and social policies) and are susceptible to "tyranny of the majority," where a slim majority can pass initiatives that are deeply harmful to a minority group. This Cardinal Social Choice System (CSCS) addresses this by:

1.  **Decoupling Issues:** It separates distinct issues into $n$ pairwise elections.
2.  **Incorporating Interest:** It gives each voter one unit of voting power to distribute across these $n$ elections. A voter who cares immensely about one issue can allocate more of their vote there, better reflecting their true priorities.

The central thesis is that such a system, by capturing preference intensity, can lead to outcomes that better maximize the average happiness of the population.

### Methodology & Key Findings

The system is modeled using a linear algebra framework where voter preferences and interests are represented as vectors. The core analytical finding of this project is the **Representative Voter Theorem**, which demonstrates that a coalition of voters can be computationally represented as a single, more powerful meta-voter, providing a rigorous way to model strategic coordination.

* **Simulation:** The repository contains Python scripts to simulate populations of voters with varying preferences and interests. This includes methods for uniform sampling from the $(n-1)$-dimensional standard simplex to generate unbiased interest distributions.
* **Optimization:** Monte Carlo methods were used to simulate strategic voting behavior and identify Nash Equilibria under different information assumptions.
* **Theoretical Result:** The primary theoretical proof shows that if voters cast their votes in proportion to their true interests, this system is guaranteed to select the outcome that maximizes the total additive happiness of the electorate.

### Tech Stack

* **Language:** Python
* **Libraries:** NumPy for vectorized computations.

### How to Use

The core logic is contained in `votingClasses.py`, which defines the `Voter` and `Population` classes. To run a simulation:

1.  Import the necessary classes from `votingClasses.py`.
2.  Instantiate a `Population` object with a specified number of voters and elections.
3.  The winner of each election can be calculated by summing the cast votes and taking the sign of the result for each election index.

```python
# Example Usage
from votingClasses import Population

# Create a population of 1000 voters and 5 elections
pop = Population(num_voters=1000, num_elections=5)

# Tally the votes
tally_vector = pop.tally_votes()

# Determine the winners
# A positive value in the winner_vector at index i means 'alternative 1' won election i.
# A negative value means 'alternative -1' won.
winner_vector = [1 if x > 0 else -1 for x in tally_vector]

print(f"Winner Vector: {winner_vector}")
```
