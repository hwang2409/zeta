"""Static shell completion scripts for the zeta CLI."""

from __future__ import annotations

_INSTALL_COMMENT = "# install: zeta completion {shell} > {destination}\n"


def zsh_script() -> str:
    return _INSTALL_COMMENT.format(shell="zsh", destination="~/.zsh/completions/_zeta") + r'''#compdef zeta

_zeta() {
    local context state line
    typeset -A opt_args
    _arguments -C \
        '(-h --help)'{-h,--help}'[show help]' \
        '(-c --continue --resume --no-session)'{-c,--continue}'[resume the most recent session]' \
        '(-c --continue --resume --no-session)--resume=[resume a session]:session id:' \
        '(-c --continue --resume --no-session)--no-session[run without persistence]' \
        '--provider=[completion provider]:provider:(fake claude codex)' \
        '--model=[provider model override]:model:' \
        '--force-provider[allow provider or model overrides during resume]' \
        '--verbose[show raw stream events]' \
        '--yolo[auto-approve every tool call]' \
        '--no-yolo[force prompts even when settings enable yolo]' \
        '--token-budget=[context token budget]:tokens:' \
        '--max-turns=[tool-use loop turn cap]:turns:' \
        '(-p --print)'{-p,--print}'[run one headless turn]:prompt:' \
        '--format=[headless output format]:format:(text json)' \
        '--system-prompt=[replace the built-in system prompt]:text or @file:' \
        '--append-system-prompt=[append to the default system prompt]:text or @file:' \
        '1:command:->command' \
        '*::argument:->argument'

    case $state in
        command)
            _describe 'command' 'login serve session automation completion'
            ;;
        argument)
            case $words[2] in
                login)
                    _arguments '--provider=[OAuth provider]:provider:(anthropic codex)'
                    ;;
                serve)
                    _arguments '--socket=[Unix socket path]:path:' '--port=[localhost TCP port]:port:' '--provider=[completion provider]:provider:(fake claude codex)' '--model=[provider model]:model:' '--cwd=[working directory]:directory:_directories'
                    ;;
                completion)
                    _arguments '1:shell:(zsh bash)'
                    ;;
                session)
                    case $words[3] in
                        list) _message 'no arguments' ;;
                        rename) _arguments '1:session id:' '2:display name:' ;;
                        delete) _arguments '--force[skip confirmation]' '1:session id:' ;;
                        export) _arguments '--out=[output file]:file:_files' '1:session id:' ;;
                        *) _describe 'verb' 'list rename delete export' ;;
                    esac
                    ;;
                automation)
                    case $words[3] in
                        list|daemon) _message 'no arguments' ;;
                        show|approve|disable) _arguments '1:name:' ;;
                        import) _arguments '1:JSON path:_files' ;;
                        *) _describe 'verb' 'list show approve disable import daemon' ;;
                    esac
                    ;;
            esac
            ;;
    esac
}

compdef _zeta zeta
'''


def bash_script() -> str:
    return _INSTALL_COMMENT.format(shell="bash", destination="~/.local/share/bash-completion/completions/zeta") + r'''_zeta_completions() {
    local cur prev command verb
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    command="${COMP_WORDS[1]}"
    verb="${COMP_WORDS[2]}"
    local top_flags="-h --help --provider --model --continue -c --resume --no-session --force-provider --verbose --yolo --no-yolo --token-budget --max-turns --print -p --format --system-prompt --append-system-prompt"
    local commands="login serve session automation completion"

    if (( COMP_CWORD == 1 )); then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=( $(compgen -W "$top_flags" -- "$cur") )
        else
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
        fi
        return
    fi

    case "$command" in
        login)
            COMPREPLY=( $(compgen -W "--provider" -- "$cur") )
            ;;
        serve)
            COMPREPLY=( $(compgen -W "--socket --port --provider --model --cwd" -- "$cur") )
            ;;
        completion)
            COMPREPLY=( $(compgen -W "zsh bash" -- "$cur") )
            ;;
        session)
            if (( COMP_CWORD == 2 )); then
                COMPREPLY=( $(compgen -W "list rename delete export" -- "$cur") )
            else
                case "$verb" in
                    delete) COMPREPLY=( $(compgen -W "--force" -- "$cur") ) ;;
                    export) COMPREPLY=( $(compgen -W "--out" -- "$cur") ) ;;
                esac
            fi
            ;;
        automation)
            if (( COMP_CWORD == 2 )); then
                COMPREPLY=( $(compgen -W "list show approve disable import daemon" -- "$cur") )
            fi
            ;;
        *)
            COMPREPLY=( $(compgen -W "$top_flags $commands" -- "$cur") )
            ;;
    esac
}

complete -F _zeta_completions zeta
'''


def completion_script(shell: str) -> str:
    """Return the deterministic completion script for ``shell``."""

    if shell == "zsh":
        return zsh_script()
    if shell == "bash":
        return bash_script()
    raise ValueError(f"unsupported shell: {shell}")


__all__ = ["bash_script", "completion_script", "zsh_script"]
