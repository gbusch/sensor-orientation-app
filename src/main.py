"""
Main application entry point for the Vehicle Reference Frame Transformation system.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from data_simulator import DataSimulator
from transformation import VehicleFrameTransformer
from quality_metrics import QualityAssessment
from dashboard import run_dashboard


def generate_sample_data(duration: float = 60.0, orientation_type: str = "changing", 
                        output_dir: str = "../data"):
    """Generate sample sensor data and save to files"""
    
    print(f"Generating {duration}s of sensor data with {orientation_type} orientation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    simulator = DataSimulator(duration=duration)
    sensor_data = simulator.generate_complete_dataset(orientation_type)
    
    # Save sensor data
    sensor_file = os.path.join(output_dir, f"sensor_data_{orientation_type}_{int(duration)}s.json")
    simulator.save_dataset(sensor_data, sensor_file)
    print(f"Saved sensor data to: {sensor_file}")
    
    # Transform data
    print("Transforming data to vehicle reference frame...")
    transformer = VehicleFrameTransformer()
    results = transformer.transform_sensor_data(sensor_data)
    
    # Save transformation results
    import json
    results_data = []
    for result in results:
        results_data.append({
            'timestamp': result.timestamp,
            'vehicle_acceleration': result.vehicle_acceleration.tolist(),
            'estimated_orientation': result.estimated_orientation.tolist(),
            'quality_score': result.quality_score,
            'confidence_metrics': result.confidence_metrics
        })
    
    results_file = os.path.join(output_dir, f"transformation_results_{orientation_type}_{int(duration)}s.json")
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved transformation results to: {results_file}")
    
    # Generate quality assessment
    print("Assessing transformation quality...")
    assessor = QualityAssessment()
    quality_report = assessor.assess_transformation_quality(sensor_data, results)
    
    # Save quality report
    quality_data = {
        'overall_score': quality_report.overall_score,
        'temporal_consistency': quality_report.temporal_consistency,
        'physical_plausibility': quality_report.physical_plausibility,
        'sensor_reliability': quality_report.sensor_reliability,
        'transformation_stability': quality_report.transformation_stability,
        'detailed_metrics': quality_report.detailed_metrics,
        'recommendations': quality_report.recommendations
    }
    
    quality_file = os.path.join(output_dir, f"quality_report_{orientation_type}_{int(duration)}s.json")
    with open(quality_file, 'w') as f:
        json.dump(quality_data, f, indent=2)
    print(f"Saved quality report to: {quality_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRANSFORMATION QUALITY SUMMARY")
    print("="*60)
    print(f"Overall Quality Score: {quality_report.overall_score:.3f}")
    print(f"Temporal Consistency: {quality_report.temporal_consistency:.3f}")
    print(f"Physical Plausibility: {quality_report.physical_plausibility:.3f}")
    print(f"Sensor Reliability: {quality_report.sensor_reliability:.3f}")
    print(f"Transformation Stability: {quality_report.transformation_stability:.3f}")
    print(f"\nProcessed {len(sensor_data)} sensor samples")
    print(f"Average transformation quality: {sum(r.quality_score for r in results) / len(results):.3f}")
    
    print("\nKey Metrics:")
    for key, value in quality_report.detailed_metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(quality_report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    return sensor_data, results, quality_report


def run_batch_analysis():
    """Run batch analysis with different orientation scenarios"""
    
    print("Running batch analysis with different orientation scenarios...")
    
    scenarios = [
        ("fixed", "Fixed phone orientation (dashboard mount)"),
        ("slowly_changing", "Slowly changing orientation (phone on seat)"),
        ("changing", "Dynamic orientation changes (phone in pocket/hand)")
    ]
    
    results_summary = []
    
    for orientation_type, description in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {description}")
        print('='*60)
        
        sensor_data, results, quality_report = generate_sample_data(
            duration=30.0, 
            orientation_type=orientation_type,
            output_dir="../data/batch_analysis"
        )
        
        results_summary.append({
            'scenario': orientation_type,
            'description': description,
            'overall_quality': quality_report.overall_score,
            'temporal_consistency': quality_report.temporal_consistency,
            'physical_plausibility': quality_report.physical_plausibility,
            'sensor_reliability': quality_report.sensor_reliability,
            'transformation_stability': quality_report.transformation_stability,
            'num_samples': len(sensor_data)
        })
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("BATCH ANALYSIS SUMMARY")
    print('='*80)
    print(f"{'Scenario':<20} {'Overall':<8} {'Temporal':<8} {'Physical':<8} {'Sensor':<8} {'Stability':<8}")
    print('-'*80)
    
    for result in results_summary:
        print(f"{result['scenario']:<20} "
              f"{result['overall_quality']:<8.3f} "
              f"{result['temporal_consistency']:<8.3f} "
              f"{result['physical_plausibility']:<8.3f} "
              f"{result['sensor_reliability']:<8.3f} "
              f"{result['transformation_stability']:<8.3f}")
    
    # Save batch summary
    import json
    with open('../data/batch_analysis/summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nBatch analysis complete. Results saved to ../data/batch_analysis/")


def main():
    """Main application entry point"""
    
    parser = argparse.ArgumentParser(
        description="Vehicle Reference Frame Transformation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py dashboard                    # Run interactive dashboard
  python main.py generate --duration 60      # Generate 60s of sample data
  python main.py batch                       # Run batch analysis
  python main.py generate --orientation fixed --duration 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run interactive dashboard')
    dashboard_parser.add_argument('--host', default='0.0.0.0', help='Host address (default: 0.0.0.0)')
    dashboard_parser.add_argument('--port', type=int, default=52739, help='Port number (default: 52739)')
    dashboard_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate sample data')
    generate_parser.add_argument('--duration', type=float, default=60.0, help='Simulation duration in seconds (default: 60)')
    generate_parser.add_argument('--orientation', choices=['fixed', 'slowly_changing', 'changing'], 
                                default='changing', help='Phone orientation type (default: changing)')
    generate_parser.add_argument('--output', default='../data', help='Output directory (default: ../data)')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Run batch analysis with different scenarios')
    
    args = parser.parse_args()
    
    if args.command == 'dashboard':
        print("Starting interactive dashboard...")
        print(f"Open your browser to: http://localhost:{args.port}")
        run_dashboard(host=args.host, port=args.port, debug=args.debug)
        
    elif args.command == 'generate':
        generate_sample_data(
            duration=args.duration,
            orientation_type=args.orientation,
            output_dir=args.output
        )
        
    elif args.command == 'batch':
        run_batch_analysis()
        
    else:
        # Default to dashboard if no command specified
        print("No command specified. Starting interactive dashboard...")
        print("Use --help to see available commands")
        print(f"Open your browser to: http://localhost:52739")
        run_dashboard()


if __name__ == "__main__":
    main()