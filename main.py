
from analysis import HairlineAnalyzer
import cv2

def main():
    analyzer = HairlineAnalyzer(base_path='hair_analysis_data')
    
    try:
        results = analyzer.process_weekly_images(
            'img/week1_left.png',
            'img/week1_right.png'
        )
        # print("Analysis results:", results)
        
        analyzer.visualize_results(cv2.imread('img/week1_left.png'), results.get("left_side")["hairline_data"], results.get("left_side")["texture_metrics"], "Left")
        analyzer.visualize_results(cv2.imread('img/week1_right.png'), results.get("right_side")["hairline_data"], results.get("right_side")["texture_metrics"], "Right")

    except Exception as e:
        print(f"Error processing images: {e}")

if __name__ == "__main__":
    main()