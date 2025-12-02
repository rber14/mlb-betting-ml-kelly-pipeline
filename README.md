📂 Key Files

    build_training_data.py

        Collects and processes MLB game data into a structured dataset suitable for machine learning.

        Likely includes features such as team stats, player performance, and historical outcomes.

    mlb_game_features.py

        Defines the feature engineering logic for MLB games.

        Converts raw stats into predictive variables (e.g., batting averages, pitcher ERA, win streaks).

    train_model.py

        Trains a machine learning model on the prepared dataset.

        Could use classification (win/loss) or probability estimation for game outcomes.

    calibrated_model.py & calibration_log.py

        Focus on probability calibration (ensuring predicted win probabilities match real-world frequencies).

        Calibration is crucial in betting models because miscalibrated probabilities can lead to poor bankroll management.

    predict_bets.py

        Uses the trained and calibrated model to generate betting recommendations.

        Applies the Kelly criterion to determine optimal bet sizing based on predicted probabilities and odds.

    run_daily.sh

        A shell script to automate the daily pipeline:

            Build training data

            Train/update the model

            Predict bets for upcoming games

            Output recommendations

⚙️ What It Actually Does

    Data Pipeline: Automates the collection and transformation of MLB game data.

    Model Training: Builds predictive models to estimate win probabilities.

    Calibration: Adjusts predictions to align with real-world outcomes.

    Betting Strategy: Uses the Kelly criterion to size bets optimally, balancing risk and reward.

    Automation: Daily script ensures the system runs continuously during the MLB season.

📌 Why It Matters

    Sports Analytics: Demonstrates how machine learning can be applied to sports betting.

    Kelly Criterion: A mathematically proven strategy for maximizing bankroll growth while minimizing risk.

    Practical Use: Could be used by sports bettors or researchers studying predictive modeling in sports.

“This project demonstrates my ability to build a full ML pipeline: from raw data ingestion and feature engineering, through model training and calibration, to applying decision theory for optimal strategies. While the domain is MLB betting, the skills are directly transferable to any industry where predictive modeling and risk management are key.”
    
