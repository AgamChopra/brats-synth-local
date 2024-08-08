"""MLCube handler file"""

import typer
from infer import run_inference
import os

app = typer.Typer()


@app.command("infer")
def infer(
    data_path: str = typer.Option('data/', "--data_path"),
    output_path: str = typer.Option('predictions/', "--output_path"),
    weights: str = typer.Option('additional_files/weights.pt', "--weights"),
    parameters_file: str = typer.Option(
        'parameters.yaml', "--parameters_file"),
):
    try:
        if os.access(output_path, os.W_OK):
            print("The path is writable.")
        else:
            print("The path is not writable.")
    except Exception:
        print("##### WARNING: NO OUTPUT PATH!!! #####")

    print(data_path)
    print(output_path)
    print(weights)
    print(parameters_file)
    run_inference(data_path, output_path, weights)


@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
