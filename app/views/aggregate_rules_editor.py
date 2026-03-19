from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AggregateRulesEditor(QWidget):
    """Reusable editor for aggregate rules."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.value_column_combo = QComboBox()
        self.aggregate_function_combo = QComboBox()
        self.aggregate_function_combo.addItems(["sum", "count", "mean", "max", "min", "nunique"])
        self.output_column_input = QLineEdit()
        self.output_column_input.setPlaceholderText("Aggregated Value")
        self.clear_rules_btn = QPushButton("Clear All")
        self.add_rule_btn = QPushButton("Add Aggregate Rule")
        self.rules_list = QListWidget()
        self.rules_list.setContextMenuPolicy(Qt.CustomContextMenu)

        layout.addWidget(QLabel("Value Column"))
        layout.addWidget(self.value_column_combo)
        layout.addWidget(QLabel("Aggregate Function"))
        layout.addWidget(self.aggregate_function_combo)
        layout.addWidget(QLabel("Output Column"))
        layout.addWidget(self.output_column_input)
        layout.addWidget(self.add_rule_btn)

        header = QHBoxLayout()
        header.addWidget(QLabel("Aggregate Rules"))
        header.addStretch()
        header.addWidget(self.clear_rules_btn)
        layout.addLayout(header)
        layout.addWidget(self.rules_list)

    def reset_rules(self) -> None:
        """Clear the aggregate-rule editor and list."""
        self.output_column_input.clear()
        self.rules_list.clear()

    def show_rule_context_menu(self, position) -> None:
        """Show per-rule actions from the aggregate-rule list."""
        item = self.rules_list.itemAt(position)
        if item is None:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(self.rules_list.mapToGlobal(position))
        if chosen == delete_action:
            self.rules_list.takeItem(self.rules_list.row(item))

    def parse_rules(self) -> list[dict]:
        """Read rule lines from the list into structured specs."""
        rules: list[dict] = []
        for index in range(self.rules_list.count()):
            line = self.rules_list.item(index).text()
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 3:
                continue
            rules.append(
                {
                    "value_column": parts[0],
                    "agg_function": parts[1],
                    "output_column": parts[2],
                }
            )
        return rules

    def add_current_rule(self) -> None:
        """Append the currently selected rule inputs to the rule list."""
        rules = self.parse_rules()
        rules.append(
            {
                "value_column": self.value_column_combo.currentText(),
                "agg_function": self.aggregate_function_combo.currentText(),
                "output_column": self.output_column_input.text().strip(),
            }
        )

        self.rules_list.clear()
        for rule in rules:
            self.rules_list.addItem(f"{rule['value_column']} | {rule['agg_function']} | {rule['output_column']}")
        self.output_column_input.clear()
