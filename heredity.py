import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probabilities = dict()
    for person in people.keys():
        if person in one_gene:
            # The person has one copy of genes
            if people[person]["mother"] and people[person]["father"]:
                # The person got the gene from mother or father but not both
                probabilities[person] = 0
                # Gets good gene from mother and bad gene from father
                goodParent = people[person]["mother"]
                badParent = people[person]["father"]
                for i in range(2):
                    tempProbability = 1
                    # Good parent
                    if goodParent in one_gene:
                        tempProbability *= ((0.5*PROBS["mutation"]) +
                                            (0.5*(1-PROBS["mutation"])))
                    elif goodParent in two_genes:
                        # The parent definitely passes the gene and it doesn't mutate
                        tempProbability *= (1-PROBS["mutation"])
                    else:
                        # The parent doesn't pass the gene and it mutates
                        tempProbability *= PROBS["mutation"]

                    # Bad Parent
                    if badParent in one_gene:
                        tempProbability *= ((0.5*PROBS["mutation"]) +
                                            (0.5*(1-PROBS["mutation"])))
                    elif badParent in two_genes:
                        # The parent definitely passes the gene and it doesn't mutate
                        tempProbability *= PROBS["mutation"]
                    else:
                        # The parent doesn't pass the gene and it mutates
                        tempProbability *= (1-PROBS["mutation"])

                    probabilities[person] += tempProbability

                    # Swapping the parents for the other case
                    # Gets good gene from father, bad gene from mother
                    goodParent = people[person]["father"]
                    badParent = people[person]["mother"]
            else:
                probabilities[person] = PROBS["gene"][1]

            # Probability that the person has trait
            if person in have_trait:
                probabilities[person] *= PROBS["trait"][1][True]
            else:
                probabilities[person] *= PROBS["trait"][1][False]

        elif person in two_genes:
            # The person has two copies of genes
            if people[person]["mother"] and people[person]["father"]:
                # The person got the genes from both the parents
                probabilities[person] = 1

                # Calculating the probability for mother
                for parent in [people[person]["mother"], people[person]["father"]]:
                    if parent in one_gene:
                        # The parent passes on the gene with probably 0.5
                        # If the gene wasn't passed, it mutates
                        # Else the gene doesn't mutate
                        probabilities[person] *= (0.5 * PROBS["mutation"] +
                                                  0.5 * (1-PROBS["mutation"]))
                    elif parent in two_genes:
                        # The parent definitely passes the gene and it doesn't mutate
                        probabilities[person] *= (1-PROBS["mutation"])
                    else:
                        # The parent doesn't pass the gene and it mutates
                        probabilities[person] *= PROBS["mutation"]

            else:
                probabilities[person] = PROBS["gene"][2]
            pass

            # Probability that the person has trait
            if person in have_trait:
                probabilities[person] *= PROBS["trait"][2][True]
            else:
                probabilities[person] *= PROBS["trait"][2][False]

        else:
            # The person has zero copies of genes
            if people[person]["mother"] and people[person]["father"]:
                probabilities[person] = 1
                for parent in [people[person]["mother"], people[person]["father"]]:
                    if parent in one_gene:
                        # The gene if passed mutates, else doesn't
                        probabilities[person] *= (0.5 * PROBS["mutation"] +
                                                  0.5 * (1-PROBS["mutation"]))
                    elif parent in two_genes:
                        # The gene definetely mutates
                        probabilities[person] *= PROBS["mutation"]
                    else:
                        # The gene hasn't mutated
                        probabilities[person] *= (1-PROBS["mutation"])
            else:
                probabilities[person] = PROBS["gene"][0]

            # Probability that the person has trait
            if person in have_trait:
                probabilities[person] *= PROBS["trait"][0][True]
            else:
                probabilities[person] *= PROBS["trait"][0][False]

    # Calculating the joint probability
    jointProbability = 1
    for probability in probabilities.values():
        jointProbability *= probability

    return jointProbability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    print(probabilities)
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    sumOfProbabilities = 0
    for probability in probabilities.values():
        sumOfProbabilities += probability

    for key in probabilities.keys():
        probabilities[key] /= sumOfProbabilities


if __name__ == "__main__":
    main()
