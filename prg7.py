import numpy as np

def euclidean_distance(point1, point2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def manhattan_distance(point1, point2):
    """Compute Manhattan distance between two points."""
    return np.sum(np.abs(np.array(point1) - np.array(point2)))

def get_coordinates():
    """Get valid user input for coordinates."""
    while True:
        try:
            coords = list(map(float, input().strip().split()))
            return coords
        except ValueError:
            print("Invalid input! Please enter numeric values separated by spaces.")

def main():
    print("Enter the coordinates of two points (space-separated):")
    
    print("Point 1: ", end="")
    point1 = get_coordinates()
    
    print("Point 2: ", end="")
    point2 = get_coordinates()

    # Check if dimensions match
    if len(point1) != len(point2):
        print("Error: Points must have the same number of dimensions.")
        return

    # Calculate distances
    euclidean = euclidean_distance(point1, point2)
    manhattan = manhattan_distance(point1, point2)

    # Print results
    print("\nDistance Calculations:")
    print(f"Euclidean Distance : {euclidean:.4f}")
    print(f"Manhattan Distance : {manhattan:.4f}")

# Correct main check
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")