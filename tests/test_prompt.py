import argparse
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama import create_ollama_client
from src.prompt import create_prompt as create_tv_prompt
from src.data_types import LLMGenerateFn, TVRequest, TVResponse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some queries.")
    parser.add_argument(
        "--query", type=str, required=True, help="The actual query to test"
    )
    parser.add_argument(
        "--provider", type=str, choices=["ollama", "groq", "mock"], help="The provider to use for the query"
    )
    parser.add_argument(
        "--model", type=str, help="The model to use for the query"
    )
    
    return parser.parse_args()

async def complete_prompt(query: str, model_name: str, provider: str) -> TVResponse:
    if provider == "ollama":
        generate_llm_response: LLMGenerateFn = create_ollama_client(model_name)
    else:
        raise ValueError(f"Invalid provider: {provider}")
    
    prompt = create_tv_prompt(query)

    llm_response: TVResponse = await generate_llm_response(
        llm_request=TVRequest(query=query, prompt=prompt, as_json=True)
    )
    return llm_response

async def main():
    args = parse_arguments()
    llm_response: TVResponse = await complete_prompt(
        query=args.query, model_name=args.model, provider=args.provider
    )

    # Save response to a file in the tmp directory one level up
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    output_file = os.path.join(tmp_dir, f"query-log-{llm_response.model_provider}-{llm_response.model_name}.json")
    
    with open(output_file, "w") as f:
        f.write(llm_response.model_dump_json(indent=2))
    
    print(f"\nResponse saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())