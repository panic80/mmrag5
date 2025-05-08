#!/usr/bin/env python3
"""
RAG Retrieval Pipeline - Phase 3 Demonstration

This script demonstrates the advanced architecture and cutting-edge techniques
implemented in Phase 3 of the RAG Retrieval Pipeline improvement plan.

Features demonstrated:
- Microservices architecture with service discovery
- ColBERT token-level interaction
- SPLADE sparse vector retrieval
- Dynamic parameter tuning
- Automated strategy selection
"""
import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# Initialize console for rich output
console = Console()

# Service URLs (with defaults for local development)
SERVICE_URLS = {
    "query": os.environ.get("QUERY_SERVICE_URL", "http://localhost:8004"),
    "cache": os.environ.get("CACHE_SERVICE_URL", "http://localhost:8000"),
    "vector_search": os.environ.get("VECTOR_SERVICE_URL", "http://localhost:8001"),
    "bm25": os.environ.get("BM25_SERVICE_URL", "http://localhost:8002"),
    "fusion": os.environ.get("FUSION_SERVICE_URL", "http://localhost:8003"),
    "colbert": os.environ.get("COLBERT_SERVICE_URL", "http://localhost:8005"),
    "splade": os.environ.get("SPLADE_SERVICE_URL", "http://localhost:8006"),
    "parameter_tuner": os.environ.get("PARAMETER_TUNER_URL", "http://localhost:8007"),
    "strategy_selector": os.environ.get("STRATEGY_SELECTOR_URL", "http://localhost:8008"),
}

# Sample queries for different query types
SAMPLE_QUERIES = {
    "factual": [
        "Who was the first person to walk on the moon?",
        "What is the capital of France?",
        "When was the Declaration of Independence signed?",
        "List the major planets in our solar system."
    ],
    "conceptual": [
        "Explain how neural networks work.",
        "Why is climate change happening?",
        "How does retrieval augmented generation improve LLM outputs?",
        "Describe the process of photosynthesis."
    ],
    "entity_rich": [
        "Compare Python, JavaScript, and Rust programming languages.",
        "What are the differences between MongoDB, PostgreSQL, and Redis?",
        "Explain the relationship between Einstein, Bohr, and Heisenberg in quantum physics.",
        "Compare the impacts of Napoleon Bonaparte, Alexander the Great, and Genghis Khan."
    ],
    "complex": [
        "What are the key technological advancements that enabled the rise of modern artificial intelligence, and how have they transformed industries like healthcare and transportation?",
        "Discuss the ethical implications of using large language models for content generation and the potential mitigations for risks like misinformation and bias.",
        "Analyze the interconnections between climate change, economic policy, global migration patterns, and geopolitical stability over the next decade.",
        "How do the principles of distributed systems design influence modern cloud architecture, and what are the tradeoffs between consistency, availability, and partition tolerance?"
    ]
}

class Phase3Demo:
    """
    Demonstration of Phase 3 RAG Retrieval Pipeline improvements.
    """
    
    def __init__(self, query_collection: str = "rag_data"):
        """Initialize the demo.
        
        Args:
            query_collection: Collection to query
        """
        self.query_collection = query_collection
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def check_services(self) -> bool:
        """Check if all required services are running.
        
        Returns:
            True if all services are healthy, False otherwise
        """
        with Progress() as progress:
            task = progress.add_task("[cyan]Checking microservices...", total=len(SERVICE_URLS))
            
            service_status = {}
            all_healthy = True
            
            for service_name, url in SERVICE_URLS.items():
                try:
                    response = await self.client.get(f"{url}/health")
                    if response.status_code == 200:
                        status = response.json().get("status", "unknown")
                        if status == "healthy":
                            service_status[service_name] = "✅ Healthy"
                        else:
                            service_status[service_name] = f"⚠️ Degraded: {status}"
                            all_healthy = False
                    else:
                        service_status[service_name] = f"❌ Error: HTTP {response.status_code}"
                        all_healthy = False
                except Exception as e:
                    service_status[service_name] = f"❌ Error: {str(e)}"
                    all_healthy = False
                
                progress.update(task, advance=1)
                await asyncio.sleep(0.1)  # Small delay for visual effect
        
        # Display service status
        table = Table(title="Microservices Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        
        for service_name, status in service_status.items():
            style = "green" if "Healthy" in status else "red" if "Error" in status else "yellow"
            table.add_row(service_name, status, style=style)
        
        console.print(table)
        
        return all_healthy
    
    async def select_strategy(self, query_text: str) -> Dict[str, Any]:
        """Select the best strategy for a query using the Strategy Selector.
        
        Args:
            query_text: The query text
            
        Returns:
            Dictionary containing strategy and strategy information
        """
        console.print(f"[cyan]Selecting strategy for query:[/cyan] {query_text}")
        
        try:
            response = await self.client.post(
                f"{SERVICE_URLS['strategy_selector']}/select",
                json={
                    "request_id": f"demo_{int(time.time())}",
                    "query_text": query_text
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from Strategy Selector: {response.text}")
            
            response_data = response.json()
            strategy = response_data.get("data", {}).get("strategy", {})
            strategy_info = response_data.get("data", {}).get("strategy_info", {})
            
            return {"strategy": strategy, "strategy_info": strategy_info}
            
        except Exception as e:
            console.print(f"[red]Error selecting strategy: {str(e)}[/red]")
            # Return a default strategy
            return {
                "strategy": {
                    "primary_retriever": "hybrid",
                    "fusion_weight": 0.5,
                    "fusion_method": "linear",
                    "use_colbert": False,
                    "use_splade": False,
                    "rerank_method": "mmr"
                },
                "strategy_info": {
                    "selection_method": "fallback",
                    "error": str(e)
                }
            }
    
    async def process_query(
        self, 
        query_text: str, 
        strategy: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """Process a query using the Query Service.
        
        Args:
            query_text: The query text
            strategy: Optional strategy override
            k: Number of results to return
            
        Returns:
            Dictionary containing query results and execution information
        """
        console.print(f"[cyan]Processing query:[/cyan] {query_text}")
        
        try:
            # Prepare request data
            request_data = {
                "request_id": f"demo_{int(time.time())}",
                "query_text": query_text,
                "collection": self.query_collection,
                "k": k,
                "use_cache": True
            }
            
            # Add strategy parameters if provided
            if strategy:
                request_data["strategy_params"] = strategy
            
            # Send request to Query Service
            response = await self.client.post(
                f"{SERVICE_URLS['query']}/query",
                json=request_data
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from Query Service: {response.text}")
            
            return response.json()
            
        except Exception as e:
            console.print(f"[red]Error processing query: {str(e)}[/red]")
            return {"status": "error", "message": str(e)}
    
    def display_results(self, results: Dict[str, Any], strategy_info: Optional[Dict[str, Any]] = None):
        """Display query results in a formatted way.
        
        Args:
            results: Query results from Query Service
            strategy_info: Optional strategy selection information
        """
        if results.get("status") == "error":
            console.print(Panel(f"[red]Error: {results.get('message')}[/red]", title="Query Error"))
            return
        
        # Extract results
        results_data = results.get("data", {})
        result_items = results_data.get("results", [])
        strategy_used = results_data.get("strategy_used", {})
        execution_stats = results_data.get("execution_stats", {})
        
        # Display strategy information if available
        if strategy_info:
            # Create strategy panel
            strategy_panel = Table(show_header=False)
            strategy_panel.add_column("Property")
            strategy_panel.add_column("Value")
            
            query_type = strategy_info.get("query_type", "unknown")
            strategy_panel.add_row("Query Type", query_type)
            strategy_panel.add_row("Selection Method", strategy_info.get("selection_method", "unknown"))
            
            if "model_confidence" in strategy_info:
                confidence = f"{strategy_info['model_confidence']:.2f}"
                strategy_panel.add_row("Model Confidence", confidence)
            
            if "rule_based_strategy" in strategy_info:
                strategy_panel.add_row("Rule-Based Strategy", strategy_info["rule_based_strategy"])
            
            console.print(Panel(strategy_panel, title=f"Strategy Information"))
        
        # Display the strategy used
        strategy_table = Table(show_header=False)
        strategy_table.add_column("Component", style="cyan")
        strategy_table.add_column("Value", style="green")
        
        strategy_table.add_row("Primary Retriever", strategy_used.get("primary_retriever", "unknown"))
        
        if strategy_used.get("secondary_retriever"):
            strategy_table.add_row("Secondary Retriever", strategy_used.get("secondary_retriever"))
        
        fusion_weight = strategy_used.get("fusion_weight", 0)
        strategy_table.add_row("Vector Weight", f"{fusion_weight:.2f}")
        strategy_table.add_row("BM25 Weight", f"{1 - fusion_weight:.2f}")
        
        if strategy_used.get("fusion_method"):
            strategy_table.add_row("Fusion Method", strategy_used.get("fusion_method"))
        
        strategy_table.add_row("ColBERT", "✅ Enabled" if strategy_used.get("use_colbert") else "❌ Disabled")
        strategy_table.add_row("SPLADE", "✅ Enabled" if strategy_used.get("use_splade") else "❌ Disabled")
        
        if strategy_used.get("rerank_method"):
            strategy_table.add_row("Reranking Method", strategy_used.get("rerank_method"))
        
        console.print(Panel(strategy_table, title="Strategy Used"))
        
        # Display execution stats
        if execution_stats:
            stats_table = Table(show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Time (s)", style="green")
            
            for key, value in execution_stats.items():
                if key.endswith("_time"):
                    metric_name = key.replace("_time", "").replace("_", " ").capitalize()
                    stats_table.add_row(metric_name, f"{value:.3f}s")
            
            console.print(Panel(stats_table, title="Execution Statistics"))
        
        # Display results
        results_table = Table(title=f"Query Results ({len(result_items)} items)")
        results_table.add_column("Rank", style="cyan")
        results_table.add_column("Score", style="green")
        results_table.add_column("Document", style="white")
        
        for i, result in enumerate(result_items):
            # Format document preview
            doc_text = result.get("text", "")
            if len(doc_text) > 100:
                doc_text = doc_text[:100] + "..."
            
            # Format result row
            results_table.add_row(
                str(i + 1),
                f"{result.get('score', 0):.4f}",
                doc_text
            )
        
        console.print(results_table)
    
    async def demonstrate_query_types(self):
        """Demonstrate different query types and their strategies."""
        console.print("\n[bold cyan]Demonstrating Different Query Types[/bold cyan]")
        
        for query_type, queries in SAMPLE_QUERIES.items():
            console.print(f"\n[bold]Query Type: [magenta]{query_type.upper()}[/magenta][/bold]")
            
            # Select one query from this type
            query = queries[0]
            
            # Select strategy
            strategy_result = await self.select_strategy(query)
            strategy = strategy_result["strategy"]
            strategy_info = strategy_result["strategy_info"]
            
            # Process query
            results = await self.process_query(query, strategy)
            
            # Display results
            self.display_results(results, strategy_info)
            
            await asyncio.sleep(1)  # Pause between queries
    
    async def demonstrate_colbert(self, query: str = "How do neural networks compare to traditional algorithms?"):
        """Demonstrate ColBERT token-level interaction.
        
        Args:
            query: Query to use for demonstration
        """
        console.print("\n[bold cyan]Demonstrating ColBERT Token-Level Interaction[/bold cyan]")
        
        try:
            # Encode query with ColBERT
            response = await self.client.post(
                f"{SERVICE_URLS['colbert']}/encode_query",
                json={
                    "request_id": f"demo_{int(time.time())}",
                    "query_text": query
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from ColBERT Service: {response.text}")
            
            # Extract token information
            response_data = response.json()
            token_data = response_data.get("data", {})
            tokens = token_data.get("tokens", [])
            
            # Display token information
            tokens_table = Table(title="ColBERT Token Analysis")
            tokens_table.add_column("Token", style="cyan")
            tokens_table.add_column("Position", style="green")
            
            for i, token in enumerate(tokens):
                tokens_table.add_row(token, str(i))
            
            console.print(tokens_table)
            
            # Process query with ColBERT enabled
            strategy = {
                "primary_retriever": "vector",
                "secondary_retriever": "bm25",
                "fusion_weight": 0.7,
                "fusion_method": "linear",
                "use_colbert": True,
                "use_splade": False,
                "rerank_method": "context_aware"
            }
            
            results = await self.process_query(query, strategy)
            
            # Display results
            strategy_info = {"query_type": "conceptual", "selection_method": "demonstration"}
            self.display_results(results, strategy_info)
            
        except Exception as e:
            console.print(f"[red]Error demonstrating ColBERT: {str(e)}[/red]")
    
    async def demonstrate_splade(self, query: str = "quantum computing applications in cryptography"):
        """Demonstrate SPLADE sparse retrieval.
        
        Args:
            query: Query to use for demonstration
        """
        console.print("\n[bold cyan]Demonstrating SPLADE Sparse Retrieval[/bold cyan]")
        
        try:
            # Encode query with SPLADE
            response = await self.client.post(
                f"{SERVICE_URLS['splade']}/encode_query",
                json={
                    "request_id": f"demo_{int(time.time())}",
                    "query_text": query
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from SPLADE Service: {response.text}")
            
            # Extract sparse vector
            response_data = response.json()
            sparse_vector = response_data.get("data", {}).get("sparse_vector", {})
            
            # Display sparse vector information
            splade_table = Table(title="SPLADE Sparse Vector")
            splade_table.add_column("Term", style="cyan")
            splade_table.add_column("Weight", style="green")
            
            # Sort terms by weight
            sorted_terms = sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)
            
            # Display top 10 terms
            for term, weight in sorted_terms[:10]:
                splade_table.add_row(term, f"{weight:.4f}")
            
            console.print(splade_table)
            
            # Process query with SPLADE enabled
            strategy = {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.4,
                "fusion_method": "rrf",
                "use_colbert": False,
                "use_splade": True,
                "rerank_method": "mmr"
            }
            
            results = await self.process_query(query, strategy)
            
            # Display results
            strategy_info = {"query_type": "entity_rich", "selection_method": "demonstration"}
            self.display_results(results, strategy_info)
            
        except Exception as e:
            console.print(f"[red]Error demonstrating SPLADE: {str(e)}[/red]")
    
    async def run_demo(self):
        """Run the complete Phase 3 demonstration."""
        console.print(Panel.fit(
            "[bold cyan]RAG Retrieval Pipeline - Phase 3 Demonstration[/bold cyan]\n\n"
            "This demo showcases the advanced architecture and cutting-edge techniques "
            "implemented in Phase 3 of the RAG improvement plan:"
            "\n\n"
            "- [green]Microservices Architecture[/green]: Separate services for query analysis, vector search, BM25, fusion, etc.\n"
            "- [green]ColBERT[/green]: Token-level interaction between queries and documents\n"
            "- [green]SPLADE[/green]: Sparse lexical and expansion for improved retrieval\n"
            "- [green]Dynamic Parameter Tuning[/green]: Optimization based on historical performance\n"
            "- [green]Automated Strategy Selection[/green]: ML-based selection of retrieval strategies"
        ))
        
        # Check services
        services_ok = await self.check_services()
        if not services_ok:
            console.print("[yellow]Some services are not healthy. Demo may not work correctly.[/yellow]")
        
        # Demonstrate different query types
        await self.demonstrate_query_types()
        
        # Demonstrate ColBERT
        await self.demonstrate_colbert()
        
        # Demonstrate SPLADE
        await self.demonstrate_splade()
        
        console.print(Panel("[bold green]Phase 3 Demonstration Complete[/bold green]"))


async def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="RAG Retrieval Pipeline Phase 3 Demo")
    parser.add_argument(
        "--collection", 
        default="rag_data",
        help="Collection to query"
    )
    parser.add_argument(
        "--query",
        help="Specific query to run (if not provided, will run the full demo)"
    )
    
    args = parser.parse_args()
    
    demo = Phase3Demo(query_collection=args.collection)
    
    if args.query:
        # Run specific query only
        strategy_result = await demo.select_strategy(args.query)
        results = await demo.process_query(args.query, strategy_result["strategy"])
        demo.display_results(results, strategy_result["strategy_info"])
    else:
        # Run full demo
        await demo.run_demo()
    
    await demo.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())