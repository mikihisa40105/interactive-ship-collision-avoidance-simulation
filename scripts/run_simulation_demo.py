"""Run the representative simulation flow.

The three trained models are included under models/. Place input_data.npy
under data/ or adjust the paths before running this script.
"""

from ship_collision_model.ships import ShipType
from ship_collision_model.simulation import SimulationRepetition
from ship_collision_model.target_model import ModelIntegrator


def main():
    integrated_model = ModelIntegrator(model_dir="models", input_data_path="data/input_data.npy", plot=False)
    sim = SimulationRepetition()
    sim.simulate_and_visualize_all_tracks(
        ShipType.ANS,
        ShipType.Model,
        integrated_model,
        instance="representative",
        plot="all",
        NUM_TRIAL=100,
        OWN_LENGTH=200,
        TGT_LENGTH=200,
        TGT_DST=5,
        TGT_AZI=90,
    )


if __name__ == "__main__":
    main()
