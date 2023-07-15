import math
import numpy as np
from sympy import Symbol, Eq, solve

# Graphing capabilities
def plot_function(expression, x_min, x_max, resolution):
    x = np.linspace(x_min, x_max, resolution)
    y = eval(expression)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graph of ' + expression)
    plt.show()

# Unit conversion
def convert_unit(value, from_unit, to_unit):
    conversions = {
        'm': {
            'km': 0.001,
            'cm': 100,
            'mm': 1000
        },
        'km': {
            'm': 1000,
            'cm': 100000,
            'mm': 1000000
        },
        'cm': {
            'm': 0.01,
            'km': 0.00001,
            'mm': 10
        },
        'mm': {
            'm': 0.001,
            'km': 0.000001,
            'cm': 0.1
        }
        # Add more conversions as needed
    }

    if from_unit == to_unit:
        return value

    if from_unit in conversions and to_unit in conversions[from_unit]:
        conversion_factor = conversions[from_unit][to_unit]
        return value * conversion_factor

    return None

# Matrix operations
def add_matrices(matrix1, matrix2):
    result = np.add(matrix1, matrix2)
    return result

def subtract_matrices(matrix1, matrix2):
    result = np.subtract(matrix1, matrix2)
    return result

def multiply_matrices(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)
    return result

def calculate_determinant(matrix):
    result = np.linalg.det(matrix)
    return result

# Complex number arithmetic
def add_complex_numbers(num1, num2):
    result = num1 + num2
    return result

def subtract_complex_numbers(num1, num2):
    result = num1 - num2
    return result

def multiply_complex_numbers(num1, num2):
    result = num1 * num2
    return result

def divide_complex_numbers(num1, num2):
    result = num1 / num2
    return result

# Statistics functions
def calculate_mean(data):
    mean = np.mean(data)
    return mean

def calculate_median(data):
    median = np.median(data)
    return median

def calculate_standard_deviation(data):
    std_dev = np.std(data)
    return std_dev

def perform_regression_analysis(x_data, y_data):
    regression = np.polyfit(x_data, y_data, 1)
    return regression

# Financial calculations
def calculate_simple_interest(principal, rate, time):
    interest = (principal * rate * time) / 100
    return interest

def calculate_loan_payment(principal, rate, time):
    interest = calculate_simple_interest(principal, rate, time)
    total_payment = principal + interest
    return total_payment

# Equation solver
def solve_linear_equation(a, b):
    if a == 0:
        if b == 0:
            return 'Infinite solutions'
        else:
            return 'No solution'
    else:
        solution = -b / a
        return solution

def solve_quadratic_equation(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        return root1, root2
    elif discriminant == 0:
        root = -b / (2*a)
        return root
    else:
        return 'Complex roots'

def solve_cubic_equation(a, b, c, d):
    x = Symbol('x')
    equation = Eq(a*x**3 + b*x**2 + c*x + d, 0)
    solutions = solve(equation, x)
    return solutions

# Number base conversions
def decimal_to_binary(decimal):
    binary = bin(decimal)[2:]
    return binary

def binary_to_decimal(binary):
    decimal = int(binary, 2)
    return decimal

def decimal_to_octal(decimal):
    octal = oct(decimal)[2:]
    return octal

def octal_to_decimal(octal):
    decimal = int(octal, 8)
    return decimal

def decimal_to_hexadecimal(decimal):
    hexadecimal = hex(decimal)[2:]
    return hexadecimal

def hexadecimal_to_decimal(hexadecimal):
    decimal = int(hexadecimal, 16)
    return decimal

# Statistical distributions
def calculate_normal_distribution_probability(x, mean, std_dev):
    exponent = -(x - mean)**2 / (2 * std_dev**2)
    probability = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(exponent)
    return probability

# Numerical optimization
def gradient_descent(gradient_func, initial_value, learning_rate, num_iterations):
    current_value = initial_value
    for _ in range(num_iterations):
        gradient = gradient_func(current_value)
        current_value -= learning_rate * gradient
    return current_value

def newton_method(func, initial_value, num_iterations):
    current_value = initial_value
    for _ in range(num_iterations):
        derivative = (func(current_value + 0.0001) - func(current_value)) / 0.0001
        current_value -= func(current_value) / derivative
    return current_value

# Main loop
def main():
    while True:
        print('==============================')
        print('Command Line Calculator')
print('|||||||||')
print('salman-sk.in')
        print('==============================')
        print('1. Graphing')
        print('2. Unit Conversion')
        print('3. Matrix Operations')
        print('4. Complex Number Arithmetic')
        print('5. Statistics Functions')
        print('6. Financial Calculations')
        print('7. Equation Solver')
        print('8. Number Base Conversions')
        print('9. Statistical Distributions')
        print('10. Numerical Optimization')
        print('0. Exit')

        choice = input('Enter your choice: ')

        if choice == '1':
            expression = input('Enter a mathematical expression: ')
            x_min = float(input('Enter the minimum value of x: '))
            x_max = float(input('Enter the maximum value of x: '))
            resolution = int(input('Enter the resolution: '))
            plot_function(expression, x_min, x_max, resolution)

        elif choice == '2':
            value = float(input('Enter the value to convert: '))
            from_unit = input('Enter the current unit: ')
            to_unit = input('Enter the target unit: ')
            converted_value = convert_unit(value, from_unit, to_unit)
            if converted_value is not None:
                print(f'{value} {from_unit} = {converted_value} {to_unit}')
            else:
                print('Invalid conversion')

        elif choice == '3':
            matrix1 = np.array(eval(input('Enter the first matrix: ')))
            matrix2 = np.array(eval(input('Enter the second matrix: ')))

            print('1. Addition')
            print('2. Subtraction')
            print('3. Multiplication')
            print('4. Determinant Calculation')
            operation = input('Enter the operation: ')

            if operation == '1':
                result = add_matrices(matrix1, matrix2)
                print('Result:')
                print(result)
            elif operation == '2':
                result = subtract_matrices(matrix1, matrix2)
                print('Result:')
                print(result)
            elif operation == '3':
                result = multiply_matrices(matrix1, matrix2)
                print('Result:')
                print(result)
            elif operation == '4':
                determinant1 = calculate_determinant(matrix1)
                determinant2 = calculate_determinant(matrix2)
                print('Determinant of Matrix 1:', determinant1)
                print('Determinant of Matrix 2:', determinant2)
            else:
                print('Invalid operation')

        elif choice == '4':
            num1 = complex(input('Enter the first complex number (a + bj): '))
            num2 = complex(input('Enter the second complex number (a + bj): '))

            print('1. Addition')
            print('2. Subtraction')
            print('3. Multiplication')
            print('4. Division')
            operation = input('Enter the operation: ')

            if operation == '1':
                result = add_complex_numbers(num1, num2)
                print('Result:', result)
            elif operation == '2':
                result = subtract_complex_numbers(num1, num2)
                print('Result:', result)
            elif operation == '3':
                result = multiply_complex_numbers(num1, num2)
                print('Result:', result)
            elif operation == '4':
                result = divide_complex_numbers(num1, num2)
                print('Result:', result)
            else:
                print('Invalid operation')

        elif choice == '5':
            data = np.array(eval(input('Enter the data (as a list): ')))

            print('1. Mean')
            print('2. Median')
            print('3. Standard Deviation')
            print('4. Regression Analysis')
            operation = input('Enter the operation: ')

            if operation == '1':
                mean = calculate_mean(data)
                print('Mean:', mean)
            elif operation == '2':
                median = calculate_median(data)
                print('Median:', median)
            elif operation == '3':
                std_dev = calculate_standard_deviation(data)
                print('Standard Deviation:', std_dev)
            elif operation == '4':
                x_data = np.array(eval(input('Enter the x-data (as a list): ')))
                regression = perform_regression_analysis(x_data, data)
                print('Regression Analysis:', regression)
            else:
                print('Invalid operation')

        elif choice == '6':
            principal = float(input('Enter the principal amount: '))
            rate = float(input('Enter the interest rate: '))
            time = float(input('Enter the time period (in years): '))

            print('1. Simple Interest Calculation')
            print('2. Loan Payment Calculation')
            operation = input('Enter the operation: ')

            if operation == '1':
                interest = calculate_simple_interest(principal, rate, time)
                print('Interest:', interest)
            elif operation == '2':
                payment = calculate_loan_payment(principal, rate, time)
                print('Total Payment:', payment)
            else:
                print('Invalid operation')

        elif choice == '7':
            print('1. Linear Equation')
            print('2. Quadratic Equation')
            print('3. Cubic Equation')
            equation_type = input('Enter the type of equation: ')

            if equation_type == '1':
                a = float(input('Enter the coefficient of x: '))
                b = float(input('Enter the constant term: '))
                solution = solve_linear_equation(a, b)
                print('Solution:', solution)
            elif equation_type == '2':
                a = float(input('Enter the coefficient of x^2: '))
                b = float(input('Enter the coefficient of x: '))
                c = float(input('Enter the constant term: '))
                solution = solve_quadratic_equation(a, b, c)
                print('Solutions:', solution)
            elif equation_type == '3':
                a = float(input('Enter the coefficient of x^3: '))
                b = float(input('Enter the coefficient of x^2: '))
                c = float(input('Enter the coefficient of x: '))
                d = float(input('Enter the constant term: '))
                solution = solve_cubic_equation(a, b, c, d)
                print('Solutions:', solution)
            else:
                print('Invalid equation type')

        elif choice == '8':
            number = int(input('Enter the number: '))

            print('1. Decimal to Binary')
            print('2. Binary to Decimal')
            print('3. Decimal to Octal')
            print('4. Octal to Decimal')
            print('5. Decimal to Hexadecimal')
            print('6. Hexadecimal to Decimal')
            operation = input('Enter the operation: ')

            if operation == '1':
                binary = decimal_to_binary(number)
                print('Binary:', binary)
            elif operation == '2':
                decimal = binary_to_decimal(str(number))
                print('Decimal:', decimal)
            elif operation == '3':
                octal = decimal_to_octal(number)
                print('Octal:', octal)
            elif operation == '4':
                decimal = octal_to_decimal(str(number))
                print('Decimal:', decimal)
            elif operation == '5':
                hexadecimal = decimal_to_hexadecimal(number)
                print('Hexadecimal:', hexadecimal)
            elif operation == '6':
                decimal = hexadecimal_to_decimal(str(number))
                print('Decimal:', decimal)
            else:
                print('Invalid operation')

        elif choice == '9':
            x = float(input('Enter the value of x: '))
            mean = float(input('Enter the mean: '))
            std_dev = float(input('Enter the standard deviation: '))

            probability = calculate_normal_distribution_probability(x, mean, std_dev)
            print('Probability:', probability)

        elif choice == '10':
            print('1. Gradient Descent')
            print('2. Newton\'s Method')
            optimization_method = input('Enter the optimization method: ')

            if optimization_method == '1':
                func = lambda x: x**2 - 4*x + 3  # Example function: x^2 - 4x + 3
                initial_value = float(input('Enter the initial value: '))
                learning_rate = float(input('Enter the learning rate: '))
                num_iterations = int(input('Enter the number of iterations: '))
                optimal_value = gradient_descent(func, initial_value, learning_rate, num_iterations)
                print('Optimal Value:', optimal_value)
            elif optimization_method == '2':
                func = lambda x: x**2 - 4*x + 3  # Example function: x^2 - 4x + 3
                initial_value = float(input('Enter the initial value: '))
                num_iterations = int(input('Enter the number of iterations: '))
                optimal_value = newton_method(func, initial_value, num_iterations)
                print('Optimal Value:', optimal_value)
            else:
                print('Invalid optimization method')

        elif choice == '0':
            print('Exiting...')
            break

        else:
            print('Invalid choice')

if __name__ == '__main__':
    main()
