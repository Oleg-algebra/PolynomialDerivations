from analyze_tools import LogAnalyzer

if __name__ == "__main__":
    analyzer = LogAnalyzer()

    # Шляхи до ваших файлів
    log_file = "logs/results_log.jsonl"
    output_directory = "analysis/case_identical_results/"
    case = "IdenticalPolynomials"

    analyzer.analyze_logs(
        log_file_path=log_file,
        output_dir=output_directory,
        case_name=case
    )