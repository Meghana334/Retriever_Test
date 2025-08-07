

from pymilvus import connections, utility, exceptions as milvus_exceptions
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse your safe_connect function
def safe_connect(alias: str = "default", host: str = "localhost", port: str = "19530"):
    """
    Connect to Milvus under `alias`. If alias exists with different config,
    disconnect and reconnect with new parameters.
    """
    try:
        connections.connect(alias=alias, host=host, port=port)
    except milvus_exceptions.ConnectionConfigException as e:
        msg = str(e)
        if "Alias of" in msg and "not the same as passed in" in msg:
            connections.disconnect(alias)
            connections.connect(alias=alias, host=host, port=port)
        else:
            raise

def list_collections(alias: str = "default") -> list:
    """
    List all collections in the Milvus instance connected under given alias.
    Returns:
        A list of collection names (strings).
    """
    # Ensure connection
    logger.info(f"Connection to alias '{alias}'...")
    safe_connect(alias=alias)
    logger.info(f"Listing collections for alias '{alias}'...")
    
    try:
        # utility.list_collections returns a list of names
        names = utility.list_collections(using=alias)
        return names
    except milvus_exceptions.BaseError as e:
        print(f"Error listing collections: {e}")
        return []

def drop_collection(collection_name: str, alias: str = "default") -> bool:
    """
    Drop (delete) a collection by name.
    Returns:
        True if deletion succeeded or collection did not exist; False on error.
    """
    safe_connect(alias=alias)
    try:
        if not utility.has_collection(collection_name, using=alias):
            print(f"Collection '{collection_name}' does not exist.")
            return True
        utility.drop_collection(collection_name, using=alias)
        print(f"Collection '{collection_name}' dropped successfully.")
        return True
    except milvus_exceptions.BaseError as e:
        print(f"Error dropping collection '{collection_name}': {e}")
        return False

# # Example usage:
# if __name__ == "__main__":
#     alias = "default"
#     host = "localhost"
#     port ="19530"
#     safe_connect(alias=alias, host=host, port=port)

#     # List collections
#     cols = list_collections(alias=alias)
#     print("Existing collections:", cols)

    # # Suppose you want to delete a specific one:
    # to_delete = "my_collection_name"
    # success = drop_collection(to_delete, alias=alias)
    # if success:
    #     # verify
    #     print("After deletion, remaining:", list_collections(alias=alias))
    # else:
    #     print("Failed to delete collection:", to_delete)