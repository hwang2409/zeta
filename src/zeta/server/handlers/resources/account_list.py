"""Negotiated account catalog and login."""

from ...resource_protocol import Access, ResourceMethod

FEATURE = "account_list_idle_v1"


async def account_list(client, request_id, params, controller):
    from ...accounts import list_account_catalog
    from ...resource_protocol import require_params

    require_params(params)
    return await list_account_catalog(client, controller.runtime)


async def login_start(client, request_id, params, controller):
    from ...resource_protocol import require_params

    require_params(params, "provider", optional=("account",))
    return await client.logins.login_start(params, named_accounts=True)


METHODS = (
    ResourceMethod("account_list", Access.CONTROLLER_READ, account_list, "1.1", extends_legacy=True),
    ResourceMethod("login_start", Access.CONTROLLER_WRITE, login_start, "1.1", extends_legacy=True),
)
