"""Negotiated session-optional account catalog."""

from ...resource_protocol import Access, ResourceMethod

FEATURE = "account_list_idle_v1"


async def account_list(client, request_id, params):
    from ...accounts import list_account_catalog
    from ...resource_protocol import require_params

    require_params(params)
    return await list_account_catalog(client)


METHODS = (ResourceMethod("account_list", Access.CONTROLLER_READ, account_list, "1.1", extends_legacy=True),)
