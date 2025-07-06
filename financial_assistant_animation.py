from manimlib import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FinancialAssistantWorkflow(Scene):
    def construct(self):
        # Title
        title = Text("Financial Assistant & Budget Insights", font_size=48, color=BLUE)
        subtitle = Text("Complete Workflow Animation", font_size=24, color=GREY)
        VGroup(title, subtitle).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        VGroup(title, subtitle).move_to(UP * 2)
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(VGroup(title, subtitle)))
        self.wait(0.5)
        
        # Main workflow steps
        steps = [
            "1. Data Processing & Loading",
            "2. Customer Clustering Analysis", 
            "3. RFM Analysis",
            "4. Cash Flow Forecasting",
            "5. Spending Pattern Insights",
            "6. Product Recommendations",
            "7. AI Chatbot Integration"
        ]
        step_objects = VGroup()
        for i, step in enumerate(steps):
            step_text = Text(step, font_size=20, color=WHITE)
            step_objects.add(step_text)
        step_objects.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        step_objects.move_to(UP * 1.5 + LEFT * 3)
        self.play(Write(step_objects))
        self.wait(2)
        self.play(FadeOut(step_objects))
        self.wait(0.5)
        
        # Step 1: Data Processing
        self.show_data_processing()
        
        # Step 2: Clustering
        self.show_clustering_analysis()
        
        # Step 3: RFM Analysis
        self.show_rfm_analysis()
        
        # Step 4: Cash Flow Forecasting
        self.show_cash_flow_forecasting()
        
        # Step 5: Spending Insights
        self.show_spending_insights()
        
        # Step 6: Product Recommendations
        self.show_product_recommendations()
        
        # Step 7: Chatbot Integration
        self.show_chatbot_integration()
        
        # Final summary
        self.show_final_summary()

    def show_data_processing(self):
        # Step 1: Data Processing
        step_title = Text("Step 1: Data Processing & Loading", font_size=32, color=BLUE)
        step_title.to_edge(UP)
        explanation = Text("Processing transaction data for all users. Customer list is used for authentication. Product list contains general banking products for recommendations.", font_size=14, color=GREY, t2c={"authentication": YELLOW, "recommendations": GREEN})
        explanation.next_to(step_title, DOWN, buff=0.3)
        self.play(Write(step_title), Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))
        self.wait(0.2)

        # --- Table schema for customers.csv ---
        cust_schema_title = Text("customers.csv schema", font_size=18, color=YELLOW)
        cust_schema_title.next_to(step_title, DOWN, buff=0.7)
        cust_columns = ["customer_id", "name", "email", "phone", "address"]
        cust_table = self.create_table_schema(cust_columns, width=4.5)
        cust_table.next_to(cust_schema_title, DOWN, buff=0.15)
        cust_expl = Text("List of all customers for authentication and personalization.", font_size=12, color=GREY)
        cust_expl.next_to(cust_table, DOWN, buff=0.1)
        self.play(Write(cust_schema_title))
        self.play(FadeIn(cust_table))
        self.play(Write(cust_expl))
        self.wait(1.2)
        self.play(FadeOut(VGroup(cust_schema_title, cust_table, cust_expl)))

        # --- Table schema for products.csv ---
        prod_schema_title = Text("products.csv schema", font_size=18, color=YELLOW)
        prod_schema_title.next_to(step_title, DOWN, buff=0.7)
        prod_columns = ["product_id", "product_name", "category", "interest_rate", "features"]
        prod_table = self.create_table_schema(prod_columns, width=5.2)
        prod_table.next_to(prod_schema_title, DOWN, buff=0.15)
        prod_expl = Text("General banking products available for recommendation.", font_size=12, color=GREY)
        prod_expl.next_to(prod_table, DOWN, buff=0.1)
        self.play(Write(prod_schema_title))
        self.play(FadeIn(prod_table))
        self.play(Write(prod_expl))
        self.wait(1.2)
        self.play(FadeOut(VGroup(prod_schema_title, prod_table, prod_expl)))

        # --- Table schema for customer transaction files ---
        trans_schema_title = Text("customer transaction file schema (e.g., james_smith.csv)", font_size=18, color=YELLOW)
        trans_schema_title.next_to(step_title, DOWN, buff=0.7)
        trans_columns = ["date", "category", "debit", "credit", "balance", "description"]
        trans_table = self.create_table_schema(trans_columns, width=6.0)
        trans_table.next_to(trans_schema_title, DOWN, buff=0.15)
        trans_expl = Text("Individual customer transactions used for analysis and insights.", font_size=12, color=GREY)
        trans_expl.next_to(trans_table, DOWN, buff=0.1)
        self.play(Write(trans_schema_title))
        self.play(FadeIn(trans_table))
        self.play(Write(trans_expl))
        self.wait(1.2)
        self.play(FadeOut(VGroup(trans_schema_title, trans_table, trans_expl)))

        # --- Brief explanation of processing ---
        process_expl = Text("These tables are loaded and merged to create a unified dataset for each customer, enabling clustering, RFM analysis, forecasting, and recommendations.", font_size=13, color=GREEN)
        process_expl.next_to(step_title, DOWN, buff=1.1)
        self.play(Write(process_expl))
        self.wait(1.5)
        self.play(FadeOut(process_expl))
        self.wait(0.2)

        # Continue with the rest of the original animation...
        # Create data files representation
        data_files = VGroup()
        file_names = ["james_smith.csv", "brenda_newman.csv", "david_davis.csv", "customers.csv", "products.csv"]
        
        for i, name in enumerate(file_names):
            file_icon = Rectangle(height=0.8, width=2.5, color=GREEN, fill_opacity=0.3)
            file_text = Text(name, font_size=14, color=WHITE)
            file_text.move_to(file_icon.get_center())
            file_group = VGroup(file_icon, file_text)
            file_group.shift(RIGHT * (i - 2) * 3 + DOWN * 2)
            data_files.add(file_group)
        
        self.play(FadeIn(data_files, lag_ratio=0.2))
        self.wait(0.2)
        
        # Data processing arrows
        process_arrow = Arrow(LEFT * 6, RIGHT * 6, color=YELLOW)
        process_arrow.shift(DOWN * 0.5)
        process_text = Text("Data Processing", font_size=16, color=YELLOW)
        process_text.next_to(process_arrow, UP)
        
        self.play(ShowCreation(process_arrow))
        self.play(Write(process_text))
        self.wait(0.2)
        
        # Processed data
        processed_data = Rectangle(height=1.5, width=8, color=BLUE, fill_opacity=0.3)
        processed_data.shift(DOWN * 3)
        processed_text = Text("Processed Customer Data", font_size=18, color=WHITE)
        processed_text.move_to(processed_data.get_center())
        
        self.play(FadeIn(processed_data))
        self.play(Write(processed_text))
        self.wait(0.2)
        
        # Show data structure
        data_structure = VGroup(
            Text("Date | Category | Debit | Credit | Balance", font_size=14, color=GREY),
            Text("2024-01-01 | Food | 50.00 | 0.00 | 1000.00", font_size=12, color=WHITE),
            Text("2024-01-02 | Transport | 25.00 | 0.00 | 975.00", font_size=12, color=WHITE),
            Text("2024-01-03 | Salary | 0.00 | 2000.00 | 2975.00", font_size=12, color=WHITE)
        )
        data_structure.arrange(DOWN, buff=0.2)
        data_structure.next_to(processed_data, DOWN, buff=0.5)
        
        self.play(Write(data_structure))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(step_title, data_files, process_arrow, process_text, 
                                processed_data, processed_text, data_structure)))

    def create_table_schema(self, columns, width=5.0):
        # Helper to create a table schema diagram (single row of column headers)
        n = len(columns)
        cell_width = width / n
        table = VGroup()
        for i, col in enumerate(columns):
            rect = Rectangle(width=cell_width, height=0.5, color=WHITE, fill_opacity=0.15)
            rect.shift(RIGHT * (i - (n-1)/2) * cell_width)
            label = Text(col, font_size=12, color=WHITE)
            label.move_to(rect.get_center())
            table.add(rect, label)
        return table

    def show_clustering_analysis(self):
        # Step 2: Clustering Analysis
        step_title = Text("Step 2: Customer Clustering Analysis", font_size=32, color=BLUE)
        step_title.to_edge(UP)
        self.play(Write(step_title))
        self.wait(0.2)
        
        # Create axes first
        axes = Axes(
            x_range=(0, 5, 1),
            y_range=(0, 5, 1),
            height=4,
            width=6,
            axis_config={"color": GREY}
        )
        axes.add_coordinate_labels(font_size=12)
        axes.shift(DOWN * 0.5)  # Move down to fit in frame
        
        x_label = Text("Monetary Value", font_size=16, color=WHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.5)
        y_label = Text("Frequency", font_size=16, color=WHITE)
        y_label.next_to(axes.y_axis, LEFT, buff=0.5).rotate(PI/2)
        
        self.play(ShowCreation(axes))
        self.play(Write(x_label), Write(y_label))
        self.wait(0.2)
        
        # Create customer data points
        customers = VGroup()
        customer_data = [
            (2, 3, "High Value"), (1, 1, "Low Value"), (3, 4, "VIP"),
            (0.5, 0.5, "At Risk"), (2.5, 2.5, "Regular"), (4, 4, "VIP"),
            (1.5, 1, "Regular"), (0.8, 0.3, "At Risk"), (3.5, 3, "High Value")
        ]
        
        colors = [RED, GREEN, GOLD, ORANGE, BLUE, GOLD, BLUE, ORANGE, RED]
        
        for i, (x, y, label) in enumerate(customer_data):
            dot = Dot(radius=0.08, color=colors[i])
            dot.move_to(axes.c2p(x, y))
            customer_text = Text(label, font_size=8, color=colors[i])
            customer_text.next_to(dot, UP, buff=0.05)
            customer_group = VGroup(dot, customer_text)
            customers.add(customer_group)
        
        self.play(FadeIn(customers, lag_ratio=0.1))
        self.wait(0.2)
        
        # Show clustering process
        cluster_centers = [
            (1, 1, "At-Risk Customers", ORANGE),
            (2, 2, "Regular Customers", BLUE), 
            (3, 3, "High-Value Customers", RED),
            (4, 4, "VIP Customers", GOLD)
        ]
        
        # Create groups to store cluster elements
        cluster_elements = VGroup()
        
        for x, y, label, color in cluster_centers:
            center = Dot(radius=0.15, color=color, fill_opacity=0.8)
            center.move_to(axes.c2p(x, y))
            center_label = Text(label, font_size=10, color=color)
            center_label.next_to(center, UP, buff=0.2)
            
            self.play(FadeIn(center), Write(center_label))
            
            # Draw cluster boundaries
            circle = Circle(radius=0.8, color=color, stroke_width=2)
            circle.move_to(center.get_center())
            self.play(ShowCreation(circle))
            
            # Add all cluster elements to the group
            cluster_elements.add(center, center_label, circle)
        
        self.wait(2)
        self.wait(0.5)
        self.play(FadeOut(VGroup(step_title, axes, x_label, y_label, customers, cluster_elements)))

    def show_rfm_analysis(self):
        # Step 3: RFM Analysis (Relative)
        step_title = Text("Step 3: RFM Analysis (Relative)", font_size=32, color=BLUE)
        step_title.to_edge(UP)
        explanation = Text("RFM values (Recency, Frequency, Monetary) are calculated for each customer and compared to others to determine relative ranking.", font_size=14, color=GREY)
        explanation.next_to(step_title, DOWN, buff=0.3)
        self.play(Write(step_title), Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))
        self.wait(0.2)

        # Simulated RFM data for 5 customers
        customers = ["James", "Brenda", "David", "Megan", "Michael"]
        recency = [5, 12, 20, 7, 30]
        frequency = [45, 30, 20, 38, 10]
        monetary = [8500, 6000, 4000, 7000, 2000]
        highlight_idx = 0  # James

        # Bar chart for each RFM metric
        rfm_bars = VGroup()
        bar_width = 0.5
        bar_spacing = 1.1
        max_height = 2.0
        for i, (r, f, m) in enumerate(zip(recency, frequency, monetary)):
            # Recency (lower is better)
            r_height = max_height * (1 - (r - min(recency)) / (max(recency) - min(recency) + 1e-6))
            r_bar = Rectangle(width=bar_width, height=r_height, fill_opacity=0.8, fill_color=BLUE if i != highlight_idx else YELLOW, stroke_color=WHITE, stroke_width=1)
            r_bar.move_to(LEFT * 3 + RIGHT * i * bar_spacing + UP * r_height/2 + UP * 1.2)
            # Frequency (higher is better)
            f_height = max_height * ((f - min(frequency)) / (max(frequency) - min(frequency) + 1e-6))
            f_bar = Rectangle(width=bar_width, height=f_height, fill_opacity=0.8, fill_color=GREEN if i != highlight_idx else YELLOW, stroke_color=WHITE, stroke_width=1)
            f_bar.move_to(LEFT * 3 + RIGHT * i * bar_spacing + UP * f_height/2)
            # Monetary (higher is better)
            m_height = max_height * ((m - min(monetary)) / (max(monetary) - min(monetary) + 1e-6))
            m_bar = Rectangle(width=bar_width, height=m_height, fill_opacity=0.8, fill_color=RED if i != highlight_idx else YELLOW, stroke_color=WHITE, stroke_width=1)
            m_bar.move_to(LEFT * 3 + RIGHT * i * bar_spacing + DOWN * 1.2 + UP * m_height/2)
            rfm_bars.add(r_bar, f_bar, m_bar)
        self.play(FadeIn(rfm_bars))
        self.wait(0.2)

        # Customer names below bars
        name_labels = VGroup()
        for i, name in enumerate(customers):
            label = Text(name, font_size=10, color=YELLOW if i == highlight_idx else WHITE)
            label.move_to(LEFT * 3 + RIGHT * i * bar_spacing + DOWN * 2.1)
            name_labels.add(label)
        self.play(Write(name_labels))
        self.wait(0.2)

        # Metric labels
        r_label = Text("Recency (lower better)", font_size=10, color=BLUE)
        r_label.move_to(LEFT * 5 + UP * 1.2)
        f_label = Text("Frequency (higher better)", font_size=10, color=GREEN)
        f_label.move_to(LEFT * 5)
        m_label = Text("Monetary (higher better)", font_size=10, color=RED)
        m_label.move_to(LEFT * 5 + DOWN * 1.2)
        self.play(Write(r_label), Write(f_label), Write(m_label))
        self.wait(0.2)

        # Annotation for relative calculation
        annotation = Text("James: Top 20% for RFM (VIP)", font_size=12, color=YELLOW)
        annotation.next_to(rfm_bars, RIGHT, buff=1.2)
        self.play(Write(annotation))
        self.wait(2)
        self.wait(0.5)
        self.play(FadeOut(VGroup(step_title, rfm_bars, name_labels, r_label, f_label, m_label, annotation)))
        self.wait(0.5)

    def show_cash_flow_forecasting(self):
        # Step 4: Cash Flow Forecasting (SARIMA)
        step_title = Text("Step 4: Cash Flow Forecasting (SARIMA)", font_size=32, color=BLUE)
        step_title.to_edge(UP)
        explanation = Text("Forecasting future balances using SARIMA model based on past 5 months of transactions.", font_size=14, color=GREY)
        explanation.next_to(step_title, DOWN, buff=0.3)
        self.play(Write(step_title), Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))
        self.wait(0.2)

        # Axes for time series
        axes = Axes(
            x_range=(0, 60, 10),
            y_range=(800, 1400, 100),
            height=3.2,
            width=7,
            axis_config={"color": GREY}
        )
        axes.add_coordinate_labels(font_size=10)
        axes.shift(DOWN * 0.5)

        x_label = Text("Days", font_size=12, color=WHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Balance ($)", font_size=12, color=WHITE)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2).rotate(PI/2)

        self.play(ShowCreation(axes))
        self.play(Write(x_label), Write(y_label))
        self.wait(0.2)

        # Simulated data
        train_x = np.arange(0, 30)
        train_y = 1000 + 10 * np.sin(train_x/5) + np.cumsum(np.random.normal(0, 10, len(train_x)))
        forecast_x = np.arange(30, 60)
        forecast_y = train_y[-1] + np.cumsum(np.random.normal(5, 10, len(forecast_x)))
        ci_upper = forecast_y + 40
        ci_lower = forecast_y - 40

        # Plot training data
        train_points = [axes.c2p(x, y) for x, y in zip(train_x, train_y)]
        train_line = VMobject(color=BLUE, stroke_width=3)
        train_line.set_points_as_corners(train_points)
        self.play(ShowCreation(train_line))
        self.wait(0.2)

        # Forecast line (dashed)
        forecast_points = [axes.c2p(x, y) for x, y in zip(forecast_x, forecast_y)]
        forecast_line = DashedVMobject(VMobject().set_points_as_corners(forecast_points), num_dashes=30, color=GREEN, stroke_width=3)
        self.play(ShowCreation(forecast_line))
        self.wait(0.2)

        # Confidence interval (shaded area)
        ci_points = [axes.c2p(x, y) for x, y in zip(forecast_x, ci_upper)] + [axes.c2p(x, y) for x, y in zip(forecast_x[::-1], ci_lower[::-1])]
        ci_polygon = Polygon(*ci_points, fill_opacity=0.2, fill_color=GREEN, stroke_width=0)
        self.play(FadeIn(ci_polygon))
        self.wait(0.2)

        # Vertical split line
        split_line = DashedLine(axes.c2p(30, 800), axes.c2p(30, 1400), color=YELLOW, stroke_width=2)
        self.play(ShowCreation(split_line))
        self.wait(0.2)

        # Labels
        train_label = Text("Training Data", font_size=11, color=BLUE)
        train_label.next_to(train_line, LEFT, buff=0.2)
        forecast_label = Text("SARIMA Forecast", font_size=11, color=GREEN)
        forecast_label.next_to(forecast_line, RIGHT, buff=0.2)
        ci_label = Text("Confidence Interval", font_size=10, color=GREEN)
        ci_label.next_to(ci_polygon, UP, buff=0.1)
        split_label = Text("Forecast Start", font_size=10, color=YELLOW)
        split_label.next_to(split_line, UP, buff=0.1)
        self.play(Write(train_label), Write(forecast_label), Write(ci_label), Write(split_label))
        self.wait(2)
        self.wait(0.5)
        self.play(FadeOut(VGroup(step_title, axes, x_label, y_label, train_line, forecast_line, ci_polygon, split_line, train_label, forecast_label, ci_label, split_label)))
        self.wait(0.5)

    def show_spending_insights(self):
        # Step 5: Spending Pattern Insights
        step_title = Text("Step 5: Spending Pattern Insights", font_size=32, color=BLUE)
        step_title.to_edge(UP)
        self.play(Write(step_title))
        self.wait(0.2)
        
        # Spending categories bar chart representation (more reliable than pie chart)
        categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment", "Utilities"]
        percentages = [35, 25, 20, 15, 5]
        colors = [RED, BLUE, GREEN, YELLOW, ORANGE]
        
        # Bar chart layout
        bar_chart = VGroup()
        category_labels = VGroup()
        percentage_labels = VGroup()
        bar_width = 0.8
        bar_spacing = 1.2
        max_height = 2.0
        bars = []
        for i, (category, percentage, color) in enumerate(zip(categories, percentages, colors)):
            bar_height = (percentage / 100) * max_height
            bar = Rectangle(width=bar_width, height=bar_height, fill_opacity=0.8, fill_color=color, stroke_color=WHITE, stroke_width=1)
            bar.move_to(LEFT * 2 + RIGHT * i * bar_spacing + UP * bar_height/2)
            bars.append(bar)
            cat_label = Text(category, font_size=10, color=WHITE)
            cat_label.next_to(bar, DOWN, buff=0.18)
            pct_label = Text(f"{percentage}%", font_size=12, color=color, weight=BOLD)
            pct_label.next_to(bar, UP, buff=0.12)
            bar_chart.add(bar)
            category_labels.add(cat_label)
            percentage_labels.add(pct_label)
        # Center the bar chart group
        bar_chart_group = VGroup(bar_chart, category_labels, percentage_labels)
        bar_chart_group.arrange(DOWN, buff=0.3)
        bar_chart_group.move_to(ORIGIN + LEFT * 1.5)
        self.play(FadeIn(bar_chart))
        self.play(Write(category_labels))
        self.play(Write(percentage_labels))
        self.wait(0.2)
        
        # Insights text
        insights = VGroup(
            Text("Key Insights:", font_size=16, color=YELLOW),
            Text("• Food & Dining: 35% (Highest)", font_size=11, color=WHITE),
            Text("• Transportation: 25% of budget", font_size=11, color=WHITE),
            Text("• Shopping: 20% (Moderate)", font_size=11, color=WHITE),
            Text("• Entertainment: 15% (Reasonable)", font_size=11, color=WHITE),
            Text("• Utilities: 5% (Well managed)", font_size=11, color=WHITE)
        )
        insights.arrange(DOWN, buff=0.13, aligned_edge=LEFT)
        insights.next_to(bar_chart, RIGHT, buff=1.2)
        self.play(Write(insights))
        self.wait(0.2)
        
        # Spending trend
        trend_text = Text("Monthly Trend: Increasing food spending", font_size=12, color=GREEN)
        trend_text.next_to(bar_chart, DOWN, buff=0.5)
        self.play(Write(trend_text))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(step_title, bar_chart, category_labels, percentage_labels, insights, trend_text)))

    def show_product_recommendations(self):
        # Step 6: Product Recommendations
        step_title = Text("Step 6: Product Recommendations", font_size=32, color=BLUE)
        step_title.to_edge(UP)
        explanation = Text("Products are recommended based on each user’s spending patterns and financial profile.", font_size=14, color=GREY)
        explanation.next_to(step_title, DOWN, buff=0.3)
        self.play(Write(step_title), Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))
        self.wait(0.2)
        
        # Customer profile
        profile = VGroup(
            Text("Customer Profile", font_size=20, color=YELLOW),
            Text("James Smith", font_size=16, color=WHITE),
            Text("High Spender", font_size=14, color=GREEN),
            Text("Monthly spending: $2,500", font_size=14, color=WHITE),
            Text("Balance: $3,200", font_size=14, color=WHITE)
        )
        profile.arrange(DOWN, buff=0.3)
        profile.shift(LEFT * 4 + UP * 1)
        
        self.play(FadeIn(profile))
        self.wait(0.2)
        
        # Recommended products
        products = VGroup(
            Text("Recommended Products", font_size=20, color=YELLOW),
            Text("1. Premium Cashback Credit Card", font_size=14, color=GREEN),
            Text("   - 2% cashback on all purchases", font_size=12, color=GREY),
            Text("2. High-Yield Savings Account", font_size=14, color=GREEN),
            Text("   - 3.5% APY on savings", font_size=12, color=GREY),
            Text("3. Travel Rewards Credit Card", font_size=14, color=GREEN),
            Text("   - 3x points on travel", font_size=12, color=GREY)
        )
        products.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        products.shift(RIGHT * 4)
        
        self.play(Write(products))
        self.wait(0.2)
        
        # Recommendation reasons
        reasons = VGroup(
            Text("Why These Products?", font_size=16, color=ORANGE),
            Text("• High spending → Cashback rewards", font_size=12, color=WHITE),
            Text("• Good balance → Savings opportunity", font_size=12, color=WHITE),
            Text("• Travel spending → Travel rewards", font_size=12, color=WHITE)
        )
        reasons.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        reasons.shift(DOWN * 2)
        
        self.play(Write(reasons))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(step_title, profile, products, reasons)))

    def show_chatbot_integration(self):
        # Step 7: Chatbot Integration (LLM, Context, History)
        step_title = Text("Step 7: AI Chatbot Integration (Ollama/Gemini)", font_size=32, color=BLUE)
        step_title.to_edge(UP)
        explanation = Text("Chatbot uses context and chat history with an LLM (Ollama/Gemini) to generate personalized answers.", font_size=14, color=GREY)
        explanation.next_to(step_title, DOWN, buff=0.3)
        self.play(Write(step_title), Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))
        self.wait(0.2)

        # --- Center and shrink all elements to fit in frame ---
        # Chat window
        chat_window = Rectangle(height=2.1, width=3.2, color=WHITE, fill_opacity=0.08, stroke_width=2)
        chat_window.move_to(LEFT * 2 + DOWN * 0.2)
        chat_title = Text("Chat", font_size=12, color=YELLOW)
        chat_title.next_to(chat_window, UP, buff=0.08)
        user_msg = Text("User: How can I save more money?", font_size=9, color=WHITE)
        ai_msg = Text("AI: Based on your data, here are tips...", font_size=9, color=GREEN)
        VGroup(user_msg, ai_msg).arrange(DOWN, buff=0.13, aligned_edge=LEFT)
        VGroup(user_msg, ai_msg).next_to(chat_window.get_top(), DOWN, buff=0.18)
        chat_group = VGroup(chat_window, chat_title, user_msg, ai_msg)
        self.play(FadeIn(chat_window), Write(chat_title), Write(user_msg), Write(ai_msg))
        self.wait(0.2)

        # Context box
        context_box = Rectangle(height=1.0, width=2.2, color=BLUE, fill_opacity=0.08, stroke_width=2)
        context_box.next_to(chat_window, RIGHT, buff=0.7)
        context_title = Text("Context", font_size=10, color=BLUE)
        context_title.next_to(context_box, UP, buff=0.04)
        context_lines = [
            "Profile: James Smith",
            "RFM: 5-4-5 (Top 20%)",
            "Insights: High food spend",
            "Recs: Cashback card"
        ]
        context_texts = [Text(line, font_size=8, color=WHITE) for line in context_lines]
        context_vg = VGroup(*context_texts).arrange(DOWN, buff=0.05, aligned_edge=LEFT)
        context_vg.next_to(context_box.get_top(), DOWN, buff=0.08)
        self.play(FadeIn(context_box), Write(context_title), Write(context_vg))
        self.wait(0.2)

        # Chat history box
        history_box = Rectangle(height=0.7, width=2.2, color=GREY, fill_opacity=0.08, stroke_width=2)
        history_box.next_to(context_box, DOWN, buff=0.3)
        history_title = Text("Chat History", font_size=10, color=GREY)
        history_title.next_to(history_box, UP, buff=0.03)
        history_lines = [
            "User: Show my spending",
            "AI: Here is your chart..."
        ]
        history_texts = [Text(line, font_size=8, color=WHITE) for line in history_lines]
        history_vg = VGroup(*history_texts).arrange(DOWN, buff=0.05, aligned_edge=LEFT)
        history_vg.next_to(history_box.get_top(), DOWN, buff=0.06)
        self.play(FadeIn(history_box), Write(history_title), Write(history_vg))
        self.wait(0.2)

        # LLM brain (Ollama)
        llm_brain = Circle(radius=0.32, color=YELLOW, fill_opacity=0.15, stroke_width=2)
        llm_brain.next_to(context_box, RIGHT, buff=1.1)
        llm_label = Text("LLM: Ollama", font_size=10, color=YELLOW)
        llm_label.next_to(llm_brain, DOWN, buff=0.03)
        self.play(FadeIn(llm_brain), Write(llm_label))
        self.wait(0.2)

        # Animate arrows from context and history to LLM
        context_arrow = Arrow(context_box.get_right(), llm_brain.get_left(), color=BLUE, buff=0.05, max_tip_length_to_length_ratio=0.15)
        history_arrow = Arrow(history_box.get_right(), llm_brain.get_left() + DOWN * 0.08, color=GREY, buff=0.05, max_tip_length_to_length_ratio=0.15)
        self.play(ShowCreation(context_arrow), ShowCreation(history_arrow))
        self.wait(0.2)

        # Animate LLM generating response
        thinking = Text("Generating personalized answer...", font_size=9, color=YELLOW)
        thinking.next_to(llm_brain, UP, buff=0.07)
        self.play(Write(thinking))
        self.wait(1)
        self.play(FadeOut(thinking))
        self.wait(0.2)

        # Arrow from LLM to chat window
        response_arrow = Arrow(llm_brain.get_left(), chat_window.get_right(), color=GREEN, buff=0.05, max_tip_length_to_length_ratio=0.15)
        self.play(ShowCreation(response_arrow))
        self.wait(0.2)

        # Show new AI message in chat
        new_ai_msg = Text("AI: Set up auto-savings, use cashback card!", font_size=9, color=GREEN)
        new_ai_msg.next_to(ai_msg, DOWN, buff=0.13)
        self.play(Write(new_ai_msg))
        self.wait(2)
        self.wait(0.5)
        self.play(FadeOut(VGroup(step_title, chat_window, chat_title, user_msg, ai_msg, new_ai_msg, context_box, context_title, context_vg, history_box, history_title, history_vg, llm_brain, llm_label, context_arrow, history_arrow, response_arrow)))
        self.wait(0.5)

    def show_final_summary(self):
        # Final Summary
        summary_title = Text("Financial Assistant Workflow Complete!", font_size=36, color=GOLD)
        summary_title.to_edge(UP)
        
        summary_points = VGroup(
            Text("✓ Data processed and loaded", font_size=18, color=GREEN),
            Text("✓ Customers clustered into segments", font_size=18, color=GREEN),
            Text("✓ RFM analysis completed", font_size=18, color=GREEN),
            Text("✓ Cash flow forecast generated", font_size=18, color=GREEN),
            Text("✓ Spending insights identified", font_size=18, color=GREEN),
            Text("✓ Product recommendations created", font_size=18, color=GREEN),
            Text("✓ AI chatbot integrated", font_size=18, color=GREEN)
        )
        summary_points.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        summary_points.shift(LEFT * 2)
        
        self.play(Write(summary_title))
        self.play(Write(summary_points))
        self.wait(0.2)
        
        # Benefits
        benefits = VGroup(
            Text("Benefits:", font_size=20, color=YELLOW),
            Text("• Personalized financial insights", font_size=16, color=WHITE),
            Text("• Data-driven recommendations", font_size=16, color=WHITE),
            Text("• Automated analysis", font_size=16, color=WHITE),
            Text("• AI-powered assistance", font_size=16, color=WHITE)
        )
        benefits.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        benefits.shift(RIGHT * 4)
        
        self.play(Write(benefits))
        self.wait(0.2)
        
        # Final message
        final_message = Text("Ready to help customers make better financial decisions!", 
                           font_size=24, color=BLUE)
        final_message.shift(DOWN * 3)
        
        self.play(Write(final_message))
        self.wait(3)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(summary_title, summary_points, benefits, final_message)))


class DataProcessingAnimation(Scene):
    def construct(self):
        # Show detailed data processing animation
        title = Text("Data Processing Pipeline", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.2)
        
        # Raw data files
        raw_files = VGroup()
        file_names = ["james_smith.csv", "brenda_newman.csv", "david_davis.csv"]
        
        for i, name in enumerate(file_names):
            file_box = Rectangle(height=1, width=3, color=RED, fill_opacity=0.3)
            file_text = Text(name, font_size=12, color=WHITE)
            file_text.move_to(file_box.get_center())
            file_group = VGroup(file_box, file_text)
            file_group.shift(LEFT * 4 + UP * (2 - i * 1.5))
            raw_files.add(file_group)
        
        self.play(FadeIn(raw_files))
        self.wait(0.2)
        
        # Processing arrows
        arrows = VGroup()
        for i in range(3):
            arrow = Arrow(LEFT * 2, RIGHT * 2, color=YELLOW)
            arrow.shift(LEFT * 2 + UP * (2 - i * 1.5))
            arrows.add(arrow)
        
        self.play(ShowCreation(arrows))
        self.wait(0.2)
        
        # Data loader
        loader = Rectangle(height=4, width=2, color=BLUE, fill_opacity=0.3)
        loader.shift(RIGHT * 2)
        loader_text = Text("Data\nLoader", font_size=14, color=WHITE)
        loader_text.move_to(loader.get_center())
        loader_group = VGroup(loader, loader_text)
        
        self.play(FadeIn(loader_group))
        self.wait(0.2)
        
        # Processed data
        processed = Rectangle(height=4, width=3, color=GREEN, fill_opacity=0.3)
        processed.shift(RIGHT * 5)
        processed_text = Text("Processed\nData", font_size=14, color=WHITE)
        processed_text.move_to(processed.get_center())
        processed_group = VGroup(processed, processed_text)
        
        self.play(FadeIn(processed_group))
        self.wait(0.2)
        
        # Data structure
        structure = VGroup(
            Text("Standardized Format:", font_size=16, color=YELLOW),
            Text("Date | Category | Debit | Credit | Balance", font_size=12, color=WHITE),
            Text("2024-01-01 | Food | 50.00 | 0.00 | 1000.00", font_size=10, color=GREY),
            Text("2024-01-02 | Transport | 25.00 | 0.00 | 975.00", font_size=10, color=GREY),
            Text("2024-01-03 | Salary | 0.00 | 2000.00 | 2975.00", font_size=10, color=GREY)
        )
        structure.arrange(DOWN, buff=0.2)
        structure.shift(DOWN * 3)
        
        self.play(Write(structure))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(title, raw_files, arrows, loader_group, processed_group, structure)))


class ClusteringAnimation(Scene):
    def construct(self):
        # Show detailed clustering animation
        title = Text("Customer Clustering Process", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.2)
        
        # Feature extraction
        features = VGroup(
            Text("Feature Extraction:", font_size=20, color=YELLOW),
            Text("• Recency (days since last transaction)", font_size=14, color=WHITE),
            Text("• Frequency (number of transactions)", font_size=14, color=WHITE),
            Text("• Monetary (total amount spent)", font_size=14, color=WHITE),
            Text("• Spending volatility", font_size=14, color=WHITE),
            Text("• Balance trends", font_size=14, color=WHITE)
        )
        features.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        features.shift(LEFT * 4)
        
        self.play(Write(features))
        self.wait(0.2)
        
        # K-means clustering
        kmeans_title = Text("K-Means Clustering", font_size=20, color=GREEN)
        kmeans_title.shift(RIGHT * 4 + UP * 2)
        
        # Show clustering steps
        steps = VGroup(
            Text("1. Initialize cluster centers", font_size=14, color=WHITE),
            Text("2. Assign customers to nearest center", font_size=14, color=WHITE),
            Text("3. Update cluster centers", font_size=14, color=WHITE),
            Text("4. Repeat until convergence", font_size=14, color=WHITE)
        )
        steps.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        steps.shift(RIGHT * 4)
        
        self.play(Write(kmeans_title))
        self.play(Write(steps))
        self.wait(0.2)
        
        # Cluster results
        results = VGroup(
            Text("Cluster Results:", font_size=16, color=ORANGE),
            Text("• VIP Customers (High Value)", font_size=12, color=GOLD),
            Text("• High-Value Customers", font_size=12, color=RED),
            Text("• Regular Customers", font_size=12, color=BLUE),
            Text("• At-Risk Customers", font_size=12, color=ORANGE)
        )
        results.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        results.shift(DOWN * 2)
        
        self.play(Write(results))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(title, features, kmeans_title, steps, results)))


class RFMAnalysisAnimation(Scene):
    def construct(self):
        # Show detailed RFM analysis
        title = Text("RFM Analysis Deep Dive", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.2)
        
        # RFM explanation
        rfm_explanation = VGroup(
            Text("RFM Analysis Components:", font_size=20, color=YELLOW),
            Text("R - Recency: How recently did the customer purchase?", font_size=14, color=WHITE),
            Text("F - Frequency: How often do they purchase?", font_size=14, color=WHITE),
            Text("M - Monetary: How much do they spend?", font_size=14, color=WHITE)
        )
        rfm_explanation.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        rfm_explanation.shift(LEFT * 4)
        
        self.play(Write(rfm_explanation))
        self.wait(0.2)
        
        # Customer example
        customer_data = VGroup(
            Text("Example Customer: James Smith", font_size=18, color=GREEN),
            Text("Recency: 5 days (Excellent)", font_size=14, color=WHITE),
            Text("Frequency: 45 transactions (High)", font_size=14, color=WHITE),
            Text("Monetary: $8,500 (High)", font_size=14, color=WHITE),
            Text("RFM Score: 5-4-5", font_size=16, color=GOLD),
            Text("Segment: VIP Customer", font_size=16, color=GOLD)
        )
        customer_data.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        customer_data.shift(RIGHT * 4)
        
        self.play(Write(customer_data))
        self.wait(0.2)
        
        # Scoring system
        scoring = VGroup(
            Text("Scoring System:", font_size=16, color=ORANGE),
            Text("5: Top 20% of customers", font_size=12, color=WHITE),
            Text("4: 20-40% of customers", font_size=12, color=WHITE),
            Text("3: 40-60% of customers", font_size=12, color=WHITE),
            Text("2: 60-80% of customers", font_size=12, color=WHITE),
            Text("1: Bottom 20% of customers", font_size=12, color=WHITE)
        )
        scoring.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        scoring.shift(DOWN * 2)
        
        self.play(Write(scoring))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(title, rfm_explanation, customer_data, scoring)))


class CashFlowForecastingAnimation(Scene):
    def construct(self):
        # Show detailed cash flow forecasting
        title = Text("Cash Flow Forecasting Process", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.2)
        
        # Forecasting methods
        methods = VGroup(
            Text("Forecasting Methods:", font_size=20, color=YELLOW),
            Text("1. SARIMA Model (Advanced)", font_size=14, color=WHITE),
            Text("   - Seasonal patterns", font_size=12, color=GREY),
            Text("   - Trend analysis", font_size=12, color=GREY),
            Text("   - Confidence intervals", font_size=12, color=GREY),
            Text("2. Simple Moving Average (Fallback)", font_size=14, color=WHITE),
            Text("   - Basic trend projection", font_size=12, color=GREY)
        )
        methods.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        methods.shift(LEFT * 4)
        
        self.play(Write(methods))
        self.wait(0.2)
        
        # Model validation
        validation = VGroup(
            Text("Model Validation:", font_size=18, color=GREEN),
            Text("• MAPE (Mean Absolute Percentage Error)", font_size=14, color=WHITE),
            Text("• Validation against historical data", font_size=14, color=WHITE),
            Text("• Confidence scoring", font_size=14, color=WHITE),
            Text("• Multiple forecast periods", font_size=14, color=WHITE)
        )
        validation.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        validation.shift(RIGHT * 4)
        
        self.play(Write(validation))
        self.wait(0.2)
        
        # Forecast output
        output = VGroup(
            Text("Forecast Output:", font_size=16, color=ORANGE),
            Text("• 30-day prediction", font_size=12, color=WHITE),
            Text("• 60-day prediction", font_size=12, color=WHITE),
            Text("• Daily net flow", font_size=12, color=WHITE),
            Text("• Confidence intervals", font_size=12, color=WHITE),
            Text("• Risk assessment", font_size=12, color=WHITE)
        )
        output.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        output.shift(DOWN * 2)
        
        self.play(Write(output))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(title, methods, validation, output)))


class ChatbotIntegrationAnimation(Scene):
    def construct(self):
        # Show detailed chatbot integration
        title = Text("AI Chatbot Integration", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.2)
        
        # System architecture
        architecture = VGroup(
            Text("System Architecture:", font_size=20, color=YELLOW),
            Text("1. User Interface (Streamlit)", font_size=14, color=WHITE),
            Text("2. Context Engine", font_size=14, color=WHITE),
            Text("3. AI Model (Gemini/Ollama)", font_size=14, color=WHITE),
            Text("4. Response Generator", font_size=14, color=WHITE),
            Text("5. Data Integration", font_size=14, color=WHITE)
        )
        architecture.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        architecture.shift(LEFT * 4)
        
        self.play(Write(architecture))
        self.wait(0.2)
        
        # Context creation
        context = VGroup(
            Text("Context Creation:", font_size=18, color=GREEN),
            Text("• Customer profile", font_size=14, color=WHITE),
            Text("• Spending patterns", font_size=14, color=WHITE),
            Text("• RFM metrics", font_size=14, color=WHITE),
            Text("• Cash flow forecast", font_size=14, color=WHITE),
            Text("• Product recommendations", font_size=14, color=WHITE)
        )
        context.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        context.shift(RIGHT * 4)
        
        self.play(Write(context))
        self.wait(0.2)
        
        # AI capabilities
        capabilities = VGroup(
            Text("AI Capabilities:", font_size=16, color=ORANGE),
            Text("• Personalized advice", font_size=12, color=WHITE),
            Text("• Budget recommendations", font_size=12, color=WHITE),
            Text("• Spending analysis", font_size=12, color=WHITE),
            Text("• Product suggestions", font_size=12, color=WHITE),
            Text("• Financial education", font_size=12, color=WHITE)
        )
        capabilities.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        capabilities.shift(DOWN * 2)
        
        self.play(Write(capabilities))
        self.wait(2)
        self.wait(0.5)
        
        self.play(FadeOut(VGroup(title, architecture, context, capabilities)))


# To run specific animations:
# manimgl financial_assistant_animation.py FinancialAssistantWorkflow
# manimgl financial_assistant_animation.py DataProcessingAnimation
# manimgl financial_assistant_animation.py ClusteringAnimation
# manimgl financial_assistant_animation.py RFMAnalysisAnimation
# manimgl financial_assistant_animation.py CashFlowForecastingAnimation
# manimgl financial_assistant_animation.py ChatbotIntegrationAnimation 