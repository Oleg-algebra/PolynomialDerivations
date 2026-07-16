from analyze_tools import LogAnalyzer

if __name__ == "__main__":
    analyzer = LogAnalyzer()

    # Шляхи до ваших файлів
    log_file = "analysis/case_identical_results/rank1.txt"
    output_directory = "analysis/case_identical_results/"
    case = "IdenticalPolynomials"

    analyzer.analyze_polynomial_degrees(
        log_file_path=log_file,
        output_dir=output_directory,
        case_name=case
    )