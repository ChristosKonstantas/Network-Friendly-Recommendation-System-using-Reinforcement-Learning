# Network-Friendly-Recommendation-System-using-Reinforcement-Learning
This is a project where we see the effectiveness of some Reinforcement Learning methods in order to provide Network Friendly Recommendations (NFR) to a user and reduce the cost of providing non-cached items. 

* Environment (Content Catalogue):

    K content items. I would pick K <= 100 initially, as the code might get pretty slow otherwise.
    For every pair of items, i and j, create a random value u_ij in [0,1] that indicates how related content j is to content i. This i basically a random array U of size K-by-K (you can choose to make it symmetric or not). Assume the diagonal of this array (i.e. all elements u_ii) are equal to 0 (to avoid recommending the same item just watched).
    Assume there is a threshold u_min in [0,1] below which two contents are "irrelevant" (this is an input parameter to play around with and observe its impact). 
    Assume C out of these contents are cached (use values of C <= 0.2 K). Cached items have cost 0, and non-cached have cost 1.

* Environment (User Model):

    A user might watch multiple items, one after the other during a session.
    After a user watches a content, N = 2 new items are recommended.
    With probability q (input parameter): she ends the viewing session (i.e. does not watch another video)
    With probability 1-q, she proceeds with one more video as follows:
        If ALL N recommended are "relevant" (i.e., have higher u_ij than u_min), then
            with probability α (input parameter): the user picks one of the N recommended items (with equal probability)
            With probability 1-α: the users picks any item k from the entire catalogue of K items with probability p_k (you can assume for simplicity that p_k = 1/K....i.e. uniform).
        If at least one of the N recommendations is irrelevant, then again the users picks any item k from the entire catalogue of K items with probability p_k.

* Control variables and objective:

    For every possible item i the user might watch, a tuple of N items to recommend.
    Objective: minimize the average total cost of items watched during a session.

* Algorithms implemented:

    Value Iteration and Policy Iteration to optimize recommendations, assuming all environment parameters are known.
    Q-learning and SARSA assuming parameters α and u_min are not known (but the u_ij relevance values, and q are still known).



* Instructions:
0) Open terminal.
1) git clone https://github.com/ChristosKonstantas/Network-Friendly-Recommendation-System-using-Reinforcement-Learning.git to a folder of preference.
2) go to venv/Scripts and execute './activate'.
3) go to the directory where the .py files are and execute 'python main.py' or 'python3 main.py'.


---

**Q-Learning flowchart and state-action estimates Q**

![image](https://github.com/user-attachments/assets/02014b6d-a063-4f3c-ace2-0388db2040b7)

![image](https://github.com/user-attachments/assets/265a8487-8085-4291-920a-307e0afd36ff)

---

**Benchmark (Value Iteration)**
![image](https://github.com/user-attachments/assets/e7409350-859f-4c89-9017-3eb074e4c259)

---

[Network-Friendly Recommendation System using Reinforcement Learning report](https://github.com/ChristosKonstantas/Network-Friendly-Recommendation-System-using-Reinforcement-Learning/blob/main/NFR_Report.pdf): to study the whole implementation of [Value Iteration](https://gibberblot.github.io/rl-notes/single-agent/value-iteration.html), [Policy Iteration](https://gibberblot.github.io/rl-notes/single-agent/policy-iteration.html), [Q-learning](https://en.wikipedia.org/wiki/Q-learning) and [SARSA](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) on the general [Network Friendly Recommendations](https://arxiv.org/abs/2110.00772) problem.
