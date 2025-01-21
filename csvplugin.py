from typing import List, Optional, Union, TypedDict, TYPE_CHECKING, Annotated, Any
from io import IOBase
import pandas as pd
import asyncio
from csv import QUOTE_NONNUMERIC

from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

if TYPE_CHECKING:
    from semantic_kernel.functions.kernel_arguments import KernelArguments
    from semantic_kernel.kernel import Kernel
    from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

class CSVRow(TypedDict):
    index: int
    data: dict

class CSVPlugin:
    def __init__(self):
        self.dataframe = None

    @kernel_function(
        name="load_csv",
        description="Loads a CSV file into the plugin",
    )
    async def load_csv(
        self, 
        path: Annotated[Union[str, IOBase], "The file path or file-like object of the CSV file"]
    ) -> Annotated[bool, "True if the CSV was loaded successfully"]:
        """Loads a CSV file into the plugin."""
        try:
            if isinstance(path, (str, IOBase)):
                self.dataframe = pd.read_csv(path)
            else:
                raise ValueError(f"Expected str or file-like object, got {type(path)}")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    @kernel_function(
        name="get_data",
        description="Gets the data from the loaded CSV",
    )
    async def get_data(self) -> Annotated[Optional[List[CSVRow]], "List of rows from the CSV file"]:
        """Gets the data from the loaded CSV."""
        if self.dataframe is None:
            return None
        data = self.dataframe.to_dict(orient='records')
        return [{"index": i, "data": row} for i, row in enumerate(data)]

    @kernel_function(
        name="query_data",
        description="Queries the loaded CSV data based on a condition",
    )
    async def query_data(
        self, 
        condition: Annotated[str, "The condition to query data (e.g., 'column > 50')"]
    ) -> Annotated[Optional[List[CSVRow]], "List of rows matching the condition"]:
        """Queries the loaded CSV data based on a condition."""
        if self.dataframe is None:
            return None
        try:
            queried_data = self.dataframe.query(condition)
            data = queried_data.to_dict(orient='records')
            return [{"index": i, "data": row} for i, row in enumerate(data)]
        except Exception as e:
            print(f"Error querying data: {e}")
            return None

async def main():
    plugin = CSVPlugin()
    await plugin.load_csv('diabetes.csv')
    data = await plugin.get_data()
    #print(data)
    queried_data = await plugin.query_data("Age > 50")
    print(queried_data)

# Run the async main function
#asyncio.run(main())
