import subprocess
import os
import sys
import re
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    return project_root

def get_src_dir():
    """Get the absolute path to the src directory"""
    project_root = get_project_root()
    src_dir = project_root / "src"
    return src_dir

def run_compiler_vectorization_report():
    """Generate compiler vectorization reports using simpler compilation"""
    src_dir = get_src_dir()
    output_dir = get_project_root() / "analysis" / "vectorization_reports"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating compiler vectorization reports...")
    
    # Create a simple test file with just the kernel functions
    test_file = output_dir / "test_kernels.c"
    create_test_file(test_file)
    
    compiler_configs = {
        'scalar': ['-O2', '-fno-tree-vectorize', '-mno-sse', '-mno-avx'],
        'vectorized': ['-O3', '-march=native', '-ffast-math'],
        'avx2': ['-O3', '-mavx2', '-mfma', '-ffast-math']
    }
    
    reports = {}
    
    for config_name, flags in compiler_configs.items():
        print(f"  Generating report for {config_name}...")
        
        try:
            # Compile with vectorization report
            cmd = [
                'clang', 
                *flags,
                '-Rpass=vectorize',
                '-Rpass-missed=vectorize', 
                '-Rpass-analysis=vectorize',
                '-S',  # Generate assembly instead of object file
                '-o', str(output_dir / f'vectorization_{config_name}.s'),
                str(test_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=src_dir, timeout=30)
            reports[config_name] = result.stderr
            
            # Save the report to file
            report_file = output_dir / f'vectorization_report_{config_name}.txt'
            with open(report_file, 'w') as f:
                f.write(f"Compiler Vectorization Report - {config_name.upper()}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Command: " + ' '.join(cmd) + "\n\n")
                f.write(result.stderr)
                if result.stdout:
                    f.write("\nSTDOUT:\n" + result.stdout)
            
            print(f"    Saved to {report_file}")
            
        except subprocess.TimeoutExpired:
            error_msg = "Compilation timed out after 30 seconds"
            print(f"    {error_msg}")
            reports[config_name] = error_msg
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"    {error_msg}")
            reports[config_name] = error_msg
    
    return reports

def create_test_file(test_file_path):
    """Create a simple test file with just the kernel functions"""
    test_code = """
#include <stdlib.h>

typedef float f32;
typedef double f64;

// AXPY kernels
void axpy_scalar_f32(f32 a, f32* x, f32* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_vectorized_f32(f32 a, f32* x, f32* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_scalar_f64(f64 a, f64* x, f64* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_vectorized_f64(f64 a, f64* x, f64* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// Dot product kernels
void dot_product_scalar_f32(f32* x, f32* y, size_t n, f32* result) {
    f32 sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

void dot_product_vectorized_f32(f32* x, f32* y, size_t n, f32* result) {
    f32 sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

// Test function to ensure code is used
void test_kernels() {
    f32 x[16], y[16], result;
    f64 dx[16], dy[16];
    
    // Test all kernels to ensure they're not optimized away
    axpy_scalar_f32(2.0f, x, y, 16);
    axpy_vectorized_f32(2.0f, x, y, 16);
    axpy_scalar_f64(2.0, dx, dy, 16);
    axpy_vectorized_f64(2.0, dx, dy, 16);
    dot_product_scalar_f32(x, y, 16, &result);
    dot_product_vectorized_f32(x, y, 16, &result);
}

int main() {
    test_kernels();
    return 0;
}
"""
    
    with open(test_file_path, 'w') as f:
        f.write(test_code)
    
    return test_file_path

def analyze_vectorization_reports(reports):
    """Analyze compiler vectorization reports"""
    print("\nAnalyzing vectorization reports...")
    
    analysis_results = {}
    
    for config_name, report in reports.items():
        analysis = {
            'vectorized_loops': 0,
            'missed_optimizations': 0,
            'vectorized_functions': [],
            'missed_reasons': {},
            'vector_widths': {},
            'raw_report': report
        }
        
        if report.startswith('Error') or report.startswith('Compilation'):
            analysis_results[config_name] = analysis
            continue
        
        # Count vectorized loops
        vectorized_matches = re.findall(r'vectorized loop|loop vectorized', report, re.IGNORECASE)
        analysis['vectorized_loops'] = len(vectorized_matches)
        
        # Count missed optimizations
        missed_matches = re.findall(r'missed|not vectorized', report, re.IGNORECASE)
        analysis['missed_optimizations'] = len(missed_matches)
        
        # Extract vectorized functions
        function_matches = re.findall(r'vectorized.*in\s+(\w+)|in\s+(\w+).*vectorized', report, re.IGNORECASE)
        for match in function_matches:
            func_name = match[0] or match[1]
            if func_name and func_name not in analysis['vectorized_functions']:
                analysis['vectorized_functions'].append(func_name)
        
        # Extract reasons for missed vectorization
        missed_reasons = re.findall(r'because.*?([^\.\n]+)', report, re.IGNORECASE)
        for reason in missed_reasons:
            clean_reason = reason.strip()
            if clean_reason and len(clean_reason) > 5:  # Filter out short fragments
                analysis['missed_reasons'][clean_reason] = analysis['missed_reasons'].get(clean_reason, 0) + 1
        
        # Extract vector widths
        width_matches = re.findall(r'with\s+(\d+)\s*[x*]?\s*(\d+)?\s*(byte|bit)|(\d+)\s*(byte|bit) vector', report, re.IGNORECASE)
        for match in width_matches:
            if match[0]:  # e.g., "4 x 8 byte"
                width = f"{match[0]}x{match[1]}{match[2][0]}" if match[1] else f"{match[0]}{match[2][0]}"
            elif match[3]:  # e.g., "32 byte vector"
                width = f"{match[3]}{match[4][0]}"
            else:
                continue
            analysis['vector_widths'][width] = analysis['vector_widths'].get(width, 0) + 1
        
        analysis_results[config_name] = analysis
    
    return analysis_results

def generate_assembly_analysis():
    """Generate and analyze assembly code"""
    output_dir = get_project_root() / "analysis" / "disassembly"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating assembly analysis...")
    
    # Use the assembly files generated by the vectorization report
    assembly_files = list((get_project_root() / "analysis" / "vectorization_reports").glob("vectorization_*.s"))
    
    analysis_results = {}
    
    for asm_file in assembly_files:
        config_name = asm_file.stem.replace('vectorization_', '')
        print(f"  Analyzing assembly for {config_name}...")
        
        try:
            with open(asm_file, 'r') as f:
                assembly_code = f.read()
            
            analysis = analyze_assembly_code(assembly_code, config_name)
            
            # Save analysis
            analysis_file = output_dir / f"assembly_analysis_{config_name}.txt"
            with open(analysis_file, 'w') as f:
                f.write(f"Assembly Analysis - {config_name.upper()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"File: {asm_file}\n\n")
                f.write("VECTOR INSTRUCTIONS FOUND:\n")
                f.write("-" * 30 + "\n")
                for isa, count in analysis['vector_instructions'].items():
                    f.write(f"{isa}: {count} instructions\n")
                
                f.write(f"\nTOTAL VECTOR INSTRUCTIONS: {analysis['total_vector_instructions']}\n")
                f.write(f"TOTAL INSTRUCTIONS: {analysis['total_instructions']}\n")
                f.write(f"VECTORIZATION RATIO: {analysis['vectorization_ratio']:.1f}%\n")
            
            analysis_results[config_name] = analysis
            print(f"    Saved to {analysis_file}")
            
        except Exception as e:
            error_msg = f"Error analyzing assembly: {e}"
            print(f"    {error_msg}")
            analysis_results[config_name] = {'error': error_msg}
    
    return analysis_results

def analyze_assembly_code(assembly_code, config_name):
    """Analyze assembly code for vector instructions"""
    analysis = {
        'total_instructions': 0,
        'total_vector_instructions': 0,
        'vector_instructions': {},
        'vectorization_ratio': 0,
        'vector_sets': set()
    }
    
    # Vector instruction patterns for different ISAs
    vector_patterns = {
        'SSE': [
            r'mov(aps|apd|ups|upd|hps|hpd|lps|lpd)',
            r'addps|addpd|subps|subpd|mulps|mulpd|divps|divpd',
            r'shufps|unpcklps|unpckhps|unpcklpd|unpckhpd',
            r'cmpps|cmppd|maxps|maxpd|minps|minpd',
            r'sqrtps|sqrtpd|rsqrtps|rcpps'
        ],
        'AVX': [
            r'vmov(aps|apd|ups|upd)',
            r'vaddps|vaddpd|vsubps|vsubpd|vmulps|vmulpd|vdivps|vdivpd',
            r'vshufps|vunpcklps|vunpckhps|vunpcklpd|vunpckhpd',
            r'vcmpps|vcmppd|vmaxps|vmaxpd|vminps|vminpd',
            r'vsqrtps|vsqrtpd|vrsqrtps|vrcpps',
            r'vinsertf128|vextractf128|vperm2f128'
        ],
        'AVX2': [
            r'vfmadd(132|213|231)(ps|pd)',
            r'vbroadcast(ss|sd|s|d)',
            r'vpermd|vpermps|vpermpd',
            r'vpgather|vgatherdp|vgatherqp',
            r'vpbroadcast'
        ],
        'AVX512': [
            r'vadd[0-9]*ps|vadd[0-9]*pd',
            r'vmul[0-9]*ps|vmul[0-9]*pd',
            r'vfmadd[0-9]*ps|vfmadd[0-9]*pd',
            r'vperm[it]2[dp]',
            r'vscatter|vgather'
        ]
    }
    
    lines = assembly_code.split('\n')
    
    for line in lines:
        # Skip comments and directives
        if line.strip().startswith(('.', '#', '@')):
            continue
        
        # Count total instructions (lines with colons and tabs)
        if ':' in line and '\t' in line:
            analysis['total_instructions'] += 1
            instruction = line.split('\t')[-1].split()[0] if '\t' in line else line.split()[-1]
            
            # Check for vector instructions
            for isa, patterns in vector_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, instruction, re.IGNORECASE):
                        analysis['total_vector_instructions'] += 1
                        analysis['vector_instructions'][isa] = analysis['vector_instructions'].get(isa, 0) + 1
                        analysis['vector_sets'].add(isa)
                        break
    
    if analysis['total_instructions'] > 0:
        analysis['vectorization_ratio'] = (analysis['total_vector_instructions'] / analysis['total_instructions']) * 100
    
    return analysis

def generate_vectorization_summary(compiler_analysis, assembly_analysis):
    """Generate comprehensive vectorization verification summary"""
    output_dir = get_project_root() / "analysis" / "vectorization_reports"
    summary_file = output_dir / "vectorization_verification_summary.txt"
    
    print(f"Generating vectorization verification summary...")
    
    with open(summary_file, 'w') as f:
        f.write("SIMD Vectorization Verification Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("COMPILER VECTORIZATION REPORTS\n")
        f.write("-" * 40 + "\n\n")
        
        for config_name, analysis in compiler_analysis.items():
            f.write(f"{config_name.upper()} Configuration:\n")
            
            if 'raw_report' in analysis and analysis['raw_report'].startswith('Error'):
                f.write(f"  Error: {analysis['raw_report']}\n\n")
                continue
            
            f.write(f"  Vectorized loops: {analysis['vectorized_loops']}\n")
            f.write(f"  Missed optimizations: {analysis['missed_optimizations']}\n")
            
            if analysis['vectorized_functions']:
                f.write(f"  Vectorized functions: {', '.join(analysis['vectorized_functions'])}\n")
            
            if analysis['vector_widths']:
                f.write(f"  Vector widths detected: {', '.join([f'{k}({v})' for k, v in analysis['vector_widths'].items()])}\n")
            
            if analysis['missed_reasons']:
                f.write("  Missed vectorization reasons:\n")
                for reason, count in analysis['missed_reasons'].items():
                    f.write(f"    - {reason} ({count}x)\n")
            
            f.write("\n")
        
        '''f.write("ASSEMBLY ANALYSIS\n")
        f.write("-" * 40 + "\n\n")
        
        for config_name, analysis in assembly_analysis.items():
            if 'error' in analysis:
                f.write(f"{config_name.upper()} Configuration: {analysis['error']}\n\n")
                continue
            
            f.write(f"{config_name.upper()} Configuration:\n")
            f.write(f"  Total instructions: {analysis['total_instructions']}\n")
            f.write(f"  Vector instructions: {analysis['total_vector_instructions']}\n")
            f.write(f"  Vectorization ratio: {analysis['vectorization_ratio']:.1f}%\n")
            
            if analysis['vector_instructions']:
                f.write(f"  Vector instruction sets: {', '.join(analysis['vector_sets'])}\n")
                f.write(f"  Instruction counts by set:\n")
                for isa, count in analysis['vector_instructions'].items():
                    f.write(f"    - {isa}: {count}\n")
            
            f.write("\n")
        '''
        '''
        f.write("KEY FINDINGS\n")
        f.write("-" * 40 + "\n\n")
        
        # Generate conclusions
        vectorized_analysis = assembly_analysis.get('vectorized', {})
        scalar_analysis = assembly_analysis.get('scalar', {})
        
        if 'total_vector_instructions' in vectorized_analysis and 'total_vector_instructions' in scalar_analysis:
            vec_ratio = vectorized_analysis['vectorization_ratio']
            scalar_ratio = scalar_analysis['vectorization_ratio']
            
            f.write(f"Vectorization effectiveness:\n")
            f.write(f"  Scalar config: {scalar_ratio:.1f}% vector instructions\n")
            f.write(f"  Vectorized config: {vec_ratio:.1f}% vector instructions\n")
            
            if vec_ratio > scalar_ratio * 2:
                f.write("  Significant vectorization improvement detected\n")
            else:
                f.write("  Limited vectorization improvement\n")
        '''
        f.write("\nVECTORIZATION VERIFICATION:\n")
        f.write("-" * 30 + "\n")
        
        # Check evidence
        evidence_found = False
        
        # Check compiler reports
        for config_name, analysis in compiler_analysis.items():
            if analysis['vectorized_loops'] > 0:
                f.write(f"Compiler reports vectorized loops in {config_name} config\n")
                evidence_found = True
        
        # Check assembly
        for config_name, analysis in assembly_analysis.items():
            if not isinstance(analysis, dict) or 'error' in analysis:
                continue
            if analysis['total_vector_instructions'] > 10:
                f.write(f"Vector instructions found in {config_name} assembly\n")
                evidence_found = True
            if 'AVX' in analysis.get('vector_sets', []) or 'AVX2' in analysis.get('vector_sets', []):
                f.write(f"Advanced vector extensions detected in {config_name}\n")
                evidence_found = True
        
        if not evidence_found:
            f.write("Limited evidence of vectorization found\n")
            f.write("  Possible reasons:\n")
            f.write("  - Compiler flags may not be enabling vectorization\n")
            f.write("  - Functions may be too small to vectorize\n")
            f.write("  - Memory dependencies may prevent vectorization\n")
    
    print(f"Summary saved to {summary_file}")
    return summary_file

def main():
    """Main function to run all vectorization verification"""
    print("Starting vectorization verification...")
    print("=" * 50)
    
    # Generate compiler reports
    compiler_reports = run_compiler_vectorization_report()
    compiler_analysis = analyze_vectorization_reports(compiler_reports)
    
    # Generate assembly analysis
    assembly_analysis = generate_assembly_analysis()
    
    # Generate summary
    summary_file = generate_vectorization_summary(compiler_analysis, assembly_analysis)
    
    print(f"\nVectorization verification completed!")
    print(f"Summary report: {summary_file}")
    print("\nFiles generated:")
    print("1. Compiler vectorization reports in analysis/vectorization_reports/")
    print("2. Assembly analysis in analysis/disassembly/")
    print("3. Comprehensive summary with evidence of SIMD")

if __name__ == "__main__":
    main()