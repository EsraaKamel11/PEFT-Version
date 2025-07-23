def create_adversarial_examples():
    # Generate tricky questions for evaluation
    return [
        {"question": "Is charging always free in Berlin?", "answer": "No, most stations require payment."},
        {"question": "Can I use any cable for EV charging?", "answer": "No, you need a compatible cable for your vehicle and station."},
        {"question": "What happens if I overcharge my EV?", "answer": "Modern EVs have protections, but it's best to follow manufacturer guidelines."}
    ] 