"""
{
        "strArguments": "",
        "intTrustScore": None,
        "strModel": "Ontology",
        "boolCounterArgument": False
    }

"""
argument_examples = [
    {
        "strArguments": "There is no conclusive scientific evidence linking capsaicin consumption to hair loss in humans.",
        "intTrustScore": 9,
        "strModel": "LLM",
        "boolCounterArgument": True
    },
    {
        "strArguments": "A study found that mice fed a diet rich in capsaicin experienced hair loss and alopecia.",
        "intTrustScore": 5,
        "strModel": "LLM",
        "boolCounterArgument": False
    },
     {
        "strArguments": "Some people claim that eating spicy food has caused their hair loss, although this is anecdotal evidence and not scientifically proven.",
        "intTrustScore": 2,
        "strModel": "LLM",
        "boolCounterArgument": False
    },
     {
        "strArguments": "A review of 15 studies on capsaicin and its effects on the body found no significant correlation with hair loss. ",
        "intTrustScore": 8,
        "strModel": "LLM",
        "boolCounterArgument": True
    },
     {
        "strArguments": "No results are found connecting spicy food and hair loss",
        "intTrustScore": None,
        "strModel": "Ontology",
        "boolCounterArgument": False
    }
]